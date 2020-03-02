# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import random
import shutil
import sys
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize

import iep.utils as utils
import iep.programs
from iep.data import ClevrDataset, ClevrDataLoader
from iep.preprocess import tokenize, encode
import pickle


parser = argparse.ArgumentParser()

parser.add_argument('--result_output_path', type=str)
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_refexp_h5', default='data/val_refexps.h5')
parser.add_argument('--input_features_h5', default='data-ssd/val_features.h5')
parser.add_argument('--use_gt_programs', default=0, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--refexp', default=None)
parser.add_argument('--image', default=None)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)


def main(args):
  print()
  model = None
  if args.baseline_model is not None:
    print('Loading baseline model from ', args.baseline_model)
    model, _ = utils.load_baseline(args.baseline_model)
    if args.vocab_json is not None:
      new_vocab = utils.load_vocab(args.vocab_json)
      model.rnn.expand_vocab(new_vocab['refexp_token_to_idx'])
  elif args.program_generator is not None and args.execution_engine is not None:
    print('Loading program generator from ', args.program_generator)
    program_generator, _ = utils.load_program_generator(args.program_generator)
    print('Loading execution engine from ', args.execution_engine)
    execution_engine, _ = utils.load_execution_engine(args.execution_engine,verbose=False)
    if args.vocab_json is not None:
      new_vocab = utils.load_vocab(args.vocab_json)
      program_generator.expand_encoder_vocab(new_vocab['refexp_token_to_idx'])
    model = (program_generator, execution_engine)
  else:
    print('Must give either --baseline_model or --program_generator and --execution_engine')
    return

  if args.refexp is not None and args.image is not None:
    run_single_example(args, model)
  else:
    vocab = load_vocab(args)
    loader_kwargs = {
      'refexp_h5': args.input_refexp_h5,
      'feature_h5': args.input_features_h5,
      'vocab': vocab,
      'batch_size': args.batch_size,
    }
    if args.num_samples is not None and args.num_samples > 0:
      loader_kwargs['max_samples'] = args.num_samples
    if args.family_split_file is not None:
      with open(args.family_split_file, 'r') as f:
        loader_kwargs['refexp_families'] = json.load(f)
    with ClevrDataLoader(**loader_kwargs) as loader:
      run_batch(args, model, loader)


def load_vocab(args):
  path = None
  if args.baseline_model is not None:
    path = args.baseline_model
  elif args.execution_engine is not None:
    path = args.execution_engine
  elif args.program_generator is not None:
    path = args.program_generator
  return utils.load_cpu(path)['vocab']


def run_single_example(args, model):
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor

  # Build the CNN to use for feature extraction
  print('Loading CNN for feature extraction')
  cnn = build_cnn(args, dtype)

  # Load and preprocess the image
  img_size = (args.image_height, args.image_width)
  img = imread(args.image, mode='RGB')
  img = imresize(img, img_size, interp='bicubic')
  img = img.transpose(2, 0, 1)[None]
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
  img = (img.astype(np.float32) / 255.0 - mean) / std

  # Use CNN to extract features for the image
  img_var = Variable(torch.FloatTensor(img).type(dtype), volatile=True)
  feats_var = cnn(img_var)

  # Tokenize the refexp
  vocab = load_vocab(args)
  refexp_tokens = tokenize(args.refexp,
                      punct_to_keep=[';', ','],
                      punct_to_remove=['?', '.'])
  refexp_encoded = encode(refexp_tokens,
                       vocab['refexp_token_to_idx'],
                       allow_unk=True)
  refexp_encoded = torch.LongTensor(refexp_encoded).view(1, -1)
  refexp_encoded = refexp_encoded.type(dtype).long()
  refexp_var = Variable(refexp_encoded, volatile=True)

  # Run the model
  print('Running the model\n')
  scores = None
  predicted_program = None
  if type(model) is tuple:
    program_generator, execution_engine = model
    program_generator.type(dtype)
    execution_engine.type(dtype)
    predicted_program = program_generator.reinforce_sample(
                          refexp_var,
                          temperature=args.temperature,
                          argmax=(args.sample_argmax == 1))
    scores = execution_engine(feats_var, predicted_program)
  else:
    model.type(dtype)
    scores = model(refexp_var, feats_var)

  # Print results
  _, predicted_answer_idx = scores.data.cpu()[0].max(dim=0)
  predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx[0]]

  print('Question: "%s"' % args.refexp)
  print('Predicted answer: ', predicted_answer)

  if predicted_program is not None:
    print()
    print('Predicted program:')
    program = predicted_program.data.cpu()[0]
    num_inputs = 1
    for fn_idx in program:
      fn_str = vocab['program_idx_to_token'][fn_idx]
      num_inputs += iep.programs.get_num_inputs(fn_str) - 1
      print(fn_str)
      if num_inputs == 0:
        break


def build_cnn(args, dtype):
  if not hasattr(torchvision.models, args.cnn_model):
    raise ValueError('Invalid model "%s"' % args.cnn_model)
  if not 'resnet' in args.cnn_model:
    raise ValueError('Feature extraction only supports ResNets')
  whole_cnn = getattr(torchvision.models, args.cnn_model)(pretrained=True)
  layers = [
    whole_cnn.conv1,
    whole_cnn.bn1,
    whole_cnn.relu,
    whole_cnn.maxpool,
  ]
  for i in range(args.cnn_model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(whole_cnn, name))
  cnn = torch.nn.Sequential(*layers)
  cnn.type(dtype)
  cnn.eval()
  return cnn


def run_batch(args, model, loader):
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor
  if type(model) is tuple:
    program_generator, execution_engine = model
    run_our_model_batch(args, program_generator, execution_engine, loader, dtype)
  else:
    run_baseline_batch(args, model, loader, dtype)


def run_baseline_batch(args, model, loader, dtype):
  model.type(dtype)
  model.eval()

  all_scores, all_probs = [], []
  num_correct, num_samples = 0, 0
  for batch in loader:
    refexps, images, feats, answers, programs, program_lists = batch

    refexps_var = Variable(refexps.type(dtype).long(), volatile=True)
    feats_var = Variable(feats.type(dtype), volatile=True)
    scores = model(refexps_var, feats_var)
    probs = F.softmax(scores)

    _, preds = scores.data.cpu().max(1)
    all_scores.append(scores.data.cpu().clone())
    all_probs.append(probs.data.cpu().clone())

    num_correct += (preds == answers).sum()
    num_samples += preds.size(0)
    print('Ran %d samples' % num_samples)

  acc = float(num_correct) / num_samples
  print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

  all_scores = torch.cat(all_scores, 0)
  all_probs = torch.cat(all_probs, 0)
  if args.output_h5 is not None:
    print('Writing output to %s' % args.output_h5)
    with h5py.File(args.output_h5, 'w') as fout:
      fout.create_dataset('scores', data=all_scores.numpy())
      fout.create_dataset('probs', data=all_probs.numpy())


def run_our_model_batch(args, program_generator, execution_engine, loader, dtype):
  program_generator.type(dtype)
  program_generator.eval()
  execution_engine.type(dtype)
  execution_engine.eval()

  all_scores, all_programs = [], []
  all_probs = []
  num_correct, num_samples = 0, 0
  cum_I=0 ; cum_U=0

  ious = []
  for batch in loader:
    refexps, images, feats, answers, programs, program_lists, image_id= batch

    refexps_var = Variable(refexps.type(dtype).long(), volatile=True)
    feats_var = Variable(feats.type(dtype), volatile=True)

    programs_pred = program_generator.reinforce_sample(
                        refexps_var,
                        temperature=args.temperature,
                        argmax=(args.sample_argmax == 1))
    if args.use_gt_programs == 1:
      scores = execution_engine(feats_var, program_lists)
    else:
      scores = execution_engine(feats_var, programs_pred)

    preds = scores.clone()

    ##################
    # For Evaluation #
    ##################
    assert(answers.shape[-2:] == preds.shape[-2:])
    preds = preds.data.cpu().numpy()
    preds = preds[:, 1, :, :] > preds[:, 0, :, :]
    preds = preds.reshape([-1, 320,320])
    answers = answers.data.cpu().numpy()

    def compute_mask_IU(masks, target):
      assert(target.shape[-2:] == masks.shape[-2:])
      assert(target.shape == masks.shape)
      I = np.sum(np.logical_and(masks, target))
      U = np.sum(np.logical_or(masks, target))
      return I, U

    I, U = compute_mask_IU(preds, answers)
    for _pred, _ans in zip(preds, answers):
      _I, _U = compute_mask_IU(_pred, _ans)
      cur_IOU = _I*1.0/_U
      ious.append([_I, _U])

    cum_I += I; cum_U += U
    num_samples += preds.shape[0]
    print('Ran %d samples' % num_samples)
    msg = 'cumulative IoU = %f' % (cum_I*1.0/cum_U)
    print(msg, '\n')

  msg = 'cumulative IoU = %f' % (cum_I*1.0/cum_U)
  print(msg)
  with open(args.result_output_path, 'w') as of:
    json.dump({'ious':ious}, of)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
