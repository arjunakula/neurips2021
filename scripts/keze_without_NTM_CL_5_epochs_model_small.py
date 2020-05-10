#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,time
import os

import argparse
import json
import random
import shutil

import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py

import iep.utils as utils
import iep.preprocess
from iep.data import ClevrDataset, ClevrDataLoader, clevr_collate
from iep.models import ModuleNet, Seq2Seq, LstmModel, CnnLstmModel, CnnLstmSaModel
import copy

#-------------------------------------------Keze's modification begin----------------------------------
from polyaxon_client.tracking import get_data_paths, get_outputs_path
import subprocess

def trans_data(src, dst):
    print('rsync dir "{}" -> "{}"'.format(src, dst))
 
    if not os.path.exists(dst):
        os.makedirs(dst)
 
    cmd_line = "rsync -a {0} {1}".format(src, dst)
    subprocess.call(cmd_line.split())
#-------------------------------------------Keze's modification end----------------------------------


parser = argparse.ArgumentParser()

# Input data

parser.add_argument('--train_refexp_h5', default='data/train_refexps.h5')
parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--val_refexp_h5', default='data/val_refexps.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
        choices=['PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
        choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
        choices=['maxpool2', 'maxpool4', 'none'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--record_loss_every', type=int, default=1)
parser.add_argument('--checkpoint_every', default=10000, type=int)


def main(args):
  if args.randomize_checkpoint_path == 1:
    name, ext = os.path.splitext(args.checkpoint_path)
    num = random.randint(1, 1000000)
    args.checkpoint_path = '%s_%06d%s' % (name, num, ext)

  vocab = utils.load_vocab(args.vocab_json)

  if args.use_local_copies == 1:
    shutil.copy(args.train_refexp_h5, '/tmp/train_refexps.h5')
    shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
    shutil.copy(args.val_refexp_h5, '/tmp/val_refexps.h5')
    shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
    args.train_refexp_h5 = '/tmp/train_refexps.h5'
    args.train_features_h5 = '/tmp/train_features.h5'
    args.val_refexp_h5 = '/tmp/val_refexps.h5'
    args.val_features_h5 = '/tmp/val_features.h5'

  refexp_families = None
  if args.family_split_file is not None:
    with open(args.family_split_file, 'r') as f:
      refexp_families = json.load(f)

  train_loader_kwargs = {
    'refexp_h5': args.train_refexp_h5,
    'feature_h5': args.train_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle_train_data == 1,
    'refexp_families': refexp_families,
    'max_samples': args.num_train_samples,
    'num_workers': args.loader_num_workers,
  }
  val_loader_kwargs = {
    'refexp_h5': args.val_refexp_h5,
    'feature_h5': args.val_features_h5,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'refexp_families': refexp_families,
    'max_samples': args.num_val_samples,
    'num_workers': args.loader_num_workers,
  }

  class TLoader:  
    def __init__(self, kwargs, batch_size):
      import copy
      self.kwargs = copy.deepcopy(kwargs)
      self.batch_size = batch_size 
      #self.reset()

    def reset(self):
      import copy
      self.loader = self.get_loader(copy.deepcopy(self.kwargs), self.batch_size)

    def __iter__(self):
      return self.loader

    def __next__(self):
      #yield self.loader
      assert 1==0
      pass
    
    def get_dataset(self, kwargs):
      if 'refexp_h5' not in kwargs:
        raise ValueError('Must give refexp_h5')
      if 'feature_h5' not in kwargs:
        raise ValueError('Must give feature_h5')
      if 'vocab' not in kwargs:
        raise ValueError('Must give vocab')

      feature_h5_path = kwargs.pop('feature_h5')
      print('Reading features from ', feature_h5_path)
      _feature_h5 = h5py.File(feature_h5_path, 'r')

      _image_h5 = None
      if 'image_h5' in kwargs:
        image_h5_path = kwargs.pop('image_h5')
        print('Reading images from ', image_h5_path)
        _image_h5 = h5py.File(image_h5_path, 'r')

      vocab = kwargs.pop('vocab')
      mode = kwargs.pop('mode', 'prefix')

      refexp_families = kwargs.pop('refexp_families', None)
      max_samples = kwargs.pop('max_samples', None)
      refexp_h5_path = kwargs.pop('refexp_h5')
      image_idx_start_from = kwargs.pop('image_idx_start_from', None)
      print('Reading refexps from ', refexp_h5_path)
      _dataset = ClevrDataset(refexp_h5_path, _feature_h5, vocab, mode,
                              image_h5=_image_h5,
                              max_samples=max_samples,
                              refexp_families=refexp_families,
                              image_idx_start_from=image_idx_start_from)
      return _dataset 


    def get_loader(self, _loader_kwargs, batch_size):
      _batch_lis = []
      import copy
      _tic_time = time.time()
      cur_dataset = self.get_dataset(copy.deepcopy(_loader_kwargs))
      len_dataset = len(cur_dataset)
      for i, item in enumerate(cur_dataset):
        _batch_lis.append(item)
        if i>= len_dataset:
          yield clevr_collate(_batch_lis)
          raise StopIteration
          break
        if len(_batch_lis) == batch_size:
          _toc_time = time.time()
          yield clevr_collate(_batch_lis)
          _batch_lis.clear()
          _tic_time = time.time()

  train_loader = TLoader(train_loader_kwargs, args.batch_size)
  train_len    = len(train_loader.get_dataset(train_loader_kwargs))
  val_loader = TLoader(val_loader_kwargs, args.batch_size)
  val_len      = len(val_loader.get_dataset(val_loader_kwargs))

  train_loop(args, train_loader, train_len, val_loader, val_len)

  if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
    os.remove('/tmp/train_refexps.h5')
    os.remove('/tmp/train_features.h5')
    os.remove('/tmp/val_refexps.h5')
    os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, train_len, val_loader, val_len):
  vocab = utils.load_vocab(args.vocab_json)
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  pg_best_state, ee_best_state, baseline_best_state = None, None, None

  # Set up model
  if args.model_type == 'PG' or args.model_type == 'PG+EE':
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = torch.optim.Adam(program_generator.parameters(),
                                    lr=args.learning_rate)
    print('Here is the program generator:')
    print(program_generator)
  if args.model_type == 'EE' or args.model_type == 'PG+EE':
    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = torch.optim.Adam(execution_engine.parameters(),
                                    lr=args.learning_rate)
    print('Here is the execution engine:')
    print(execution_engine)
  if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    baseline_optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type
  loss_fn = torch.nn.CrossEntropyLoss().cuda()
  L1loss_fn = torch.nn.L1Loss().cuda()

  stats = {
    'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
    'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
    'best_val_acc': -1, 'model_t': 0,
  }
  t, epoch, reward_moving_average = 0, 0, 0

  set_mode('train', [program_generator, execution_engine, baseline_model])

  print('train_loader has %d samples' % train_len)
  print('val_loader has %d samples' % val_len)

  tic_time = time.time()
  toc_time = time.time()
  linear_iterations = [5,5,5,5,5,5,5,5,5,5,5,5,5,5] # CL_extension_for_AttnEntireQuestion_CL_5_epochs 
  #linear_iterations_2 = [3,3,3,3,3,3,3,3,3,3,3,3,3,3] # CL_extension_for_AttnEntireQuestion_CL_3_epochs
  #decay_iterations = [10,10,10,8,8,5,5,5,5,5,5,5,5,5] # CL_extension_for_AttnEntireQuestion_CL_annealing_10_to_5
  total_epochs_limit = sum(linear_iterations)
  #while t < args.num_iterations:
  while epoch < total_epochs_limit:
    epoch += 1
    print('Starting epoch %d' % epoch)
    print('Starting iterations %d' % t)
    train_loader.reset()
    val_loader.reset()

    cum_I=0 ; cum_U=0

    competency_level = 1 #from 1 to 10
    cum_diff = 0
    for km in range(0,len(linear_iterations)):
      cum_diff = cum_diff+linear_iterations[km]
      if (epoch-cum_diff) <=0:
        competency_level = km+1
        break

    for batch in train_loader:
      t += 1

      #filter batch based on difficulty level
      filtered_indices=[] # the ones to keep
      for i in range(0,batch[7].shape[0]):
        if(batch[7][i] <= competency_level):
          filtered_indices.append(i)

      if(len(filtered_indices) == 0):
        continue

      refexps_1, _, feats_1, answers_1, programs_1, __, image_id_1, curriculum_difficulty_1 = batch
      
      refexps = refexps_1[filtered_indices[0]].view(-1,refexps_1.shape[1])
      feats = feats_1[filtered_indices[0]].view(-1,feats_1.shape[1],feats_1.shape[2],feats_1.shape[3])
      answers = answers_1[filtered_indices[0]].view(-1,answers_1.shape[1],answers_1.shape[2])
      programs = programs_1[filtered_indices[0]].view(-1,programs_1.shape[1])
      image_id = image_id_1[filtered_indices[0]].view(-1)
      curriculum_difficulty = curriculum_difficulty_1[filtered_indices[0]].view(-1)

      for idx in range(1,len(filtered_indices)):
        refexps = torch.cat((refexps, refexps_1[filtered_indices[idx]].view(-1,refexps_1.shape[1])), 0)
        feats = torch.cat((feats, feats_1[filtered_indices[idx]].view(-1,feats_1.shape[1],feats_1.shape[2],feats_1.shape[3])), 0)
        answers = torch.cat((answers, answers_1[filtered_indices[idx]].view(-1,answers_1.shape[1],answers_1.shape[2])), 0)
        programs = torch.cat((programs, programs_1[filtered_indices[idx]].view(-1,programs_1.shape[1])), 0)
        image_id = torch.cat((image_id, image_id_1[filtered_indices[idx]].view(-1)), 0)
        curriculum_difficulty = torch.cat((curriculum_difficulty, curriculum_difficulty_1[filtered_indices[idx]].view(-1)), 0)
        
      # refexps = refexps[filtered_indices]
      # feats = feats[filtered_indices]
      # answers = answers[filtered_indices]
      # programs = programs[filtered_indices]
      # image_id = image_id[filtered_indices]
      # curriculum_difficulty = curriculum_difficulty[filtered_indices]

      refexps_var = Variable(refexps.cuda())
      feats_var = Variable(feats.cuda())
      answers_var = Variable(answers.cuda())
      if len(answers_var.shape) == 3:
        answers_var = answers_var.view(answers_var.shape[0], 1, answers_var.shape[1], answers_var.shape[2])
      if programs[0] is not None:
        programs_var = Variable(programs.cuda())

      reward = None
      

      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(refexps_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type == 'EE':
        # Train execution engine with ground-truth programs
        ee_optimizer.zero_grad()
        scores = execution_engine(feats_var, programs_var, refexps_var)
        preds = scores.clone()

        scores = scores.transpose(1,2).transpose(2,3).contiguous()
        scores = scores.view([-1,2]).cuda()
        _ans = answers_var.view([-1]).cuda()
        loss = loss_fn(scores, _ans)
        loss.backward()
        ee_optimizer.step()

        def compute_mask_IU(masks, target):
          assert(target.shape[-2:] == masks.shape[-2:])
          masks = masks.data.cpu().numpy()
          masks = masks[:, 1, :, :] > masks[:, 0, :, :]
          #masks = masks.reshape([args.batch_size, 320, 320])
          masks = masks.reshape([-1, 320, 320])
          target = target.data.cpu().numpy()
          print('np.sum(masks)={}'.format(np.sum(masks)))
          print('np.sum(target)={}'.format(np.sum(target)))
          I = np.sum(np.logical_and(masks, target))
          U = np.sum(np.logical_or(masks, target))
          return I, U

        I, U = compute_mask_IU(preds, answers)
        now_iou = I*1.0/U
        cum_I += I; cum_U += U
        cum_iou = cum_I*1.0/cum_U

        print_each = 10
        if t % print_each == 0:
          msg = 'now IoU = %f' % (now_iou)
          print(msg)
          msg = 'cumulative IoU = %f' % (cum_iou)
          print(msg)
        if t % print_each == 0:
          cur_time = time.time()
          since_last_print =  cur_time - toc_time
          toc_time = cur_time
          ellapsedtime = toc_time - tic_time
          iter_avr = since_last_print / (print_each+1e-5)
          batch_size = args.batch_size
          case_per_sec = print_each * 1 * batch_size / (since_last_print + 1e-6)
          estimatedleft = (args.num_iterations - t) * 1.0 * iter_avr
          print('ellapsedtime = %d, iter_avr = %f, case_per_sec = %f, estimatedleft = %f'
                % (ellapsedtime, iter_avr, case_per_sec, estimatedleft))

      elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(refexps_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(refexps_var)
        programs_pred = programs_pred.data.cpu().numpy()
        programs_pred = torch.LongTensor(programs_pred).cuda()

        scores = execution_engine(feats_var, programs_pred, refexps_var)

        preds = scores.clone()

        scores = scores.transpose(1,2).transpose(2,3).contiguous()
        scores = scores.view([-1,2]).cuda()
        _ans = answers_var.view([-1]).cuda()
        loss = loss_fn(scores, _ans)

        def compute_mask_IU(masks, target):
          assert(target.shape[-2:] == masks.shape[-2:])
          masks = masks.data.cpu().numpy()
          masks = masks[:, 1, :, :] > masks[:, 0, :, :]
          #masks = masks.reshape([args.batch_size, 320, 320])
          masks = masks.reshape([-1, 320, 320])
          target = target.data.cpu().numpy()
          print('np.sum(masks)={}'.format(np.sum(masks)))
          print('np.sum(target)={}'.format(np.sum(target)))
          I = np.sum(np.logical_and(masks, target))
          U = np.sum(np.logical_or(masks, target))
          return I, U

        I, U = compute_mask_IU(preds, answers)
        now_iou = I*1.0/U
        cum_I += I; cum_U += U
        cum_iou = cum_I*1.0/cum_U

        print_each = 10
        if t % print_each == 0:
          msg = 'now IoU = %f' % (now_iou); print(msg)
          msg = 'cumulative IoU = %f' % (cum_iou); print(msg)
        if t % print_each == 0:
          cur_time = time.time()
          since_last_print =  cur_time - toc_time
          toc_time = cur_time
          ellapsedtime = toc_time - tic_time
          iter_avr = since_last_print / (print_each+1e-5)
          batch_size = args.batch_size
          case_per_sec = print_each * 1 * batch_size / (since_last_print + 1e-6)
          estimatedleft = (args.num_iterations - t) * 1.0 * iter_avr
          print('ellapsedtime = %d, iter_avr = %f, case_per_sec = %f, estimatedleft = %f'
                % (ellapsedtime, iter_avr, case_per_sec, estimatedleft))

        def easy_compute_mask_IU(masks, target):
          assert(target.shape[-2:] == masks.shape[-2:])
          masks = masks.data.cpu().numpy()
          masks = masks[1, :, :] > masks[0, :, :]
          masks = masks.reshape([320, 320])
          target = target.data.cpu().numpy()
          assert(target.shape == masks.shape)
          I = np.sum(np.logical_and(masks, target))
          U = np.sum(np.logical_or(masks, target))
          return I, U

        now_ious = []
        for _pred, _answer in zip(preds, answers):
          _I, _U = easy_compute_mask_IU(_pred, _answer)
          if _U > 0:
            now_ious.append(_I*1.0/_U)
          else:
            now_ious.append(0.0)

        raw_reward = torch.FloatTensor(now_ious)
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          program_generator.reinforce_backward(centered_reward.cuda())
          pg_optimizer.step()

      if t % args.record_loss_every == 0:
        #-------------------------------------------Keze's modification begin-----------------------------------
        print(t, loss.data.item())
        stats['train_losses'].append(loss.data.item())
        #-------------------------------------------Keze's modification end----------------------------------

        stats['train_losses_ts'].append(t)
        if reward is not None:
          stats['train_rewards'].append(reward)

    #if t % args.checkpoint_every == 0:
    if epoch % args.checkpoint_every_epoch == 0:
      print('Checking training accuracy ... ')
      if args.model_type == 'PG':
        train_acc = check_accuracy(args, program_generator, execution_engine,
                                    baseline_model, train_loader)
      else:
        train_acc = 0.0

      print('train accuracy is', train_acc)
      print('Checking validation accuracy ...')
      if args.model_type == 'PG':
        val_acc = check_accuracy(args, program_generator, execution_engine,
                                baseline_model, val_loader)
      else:
        val_acc = 0.0

      print('val accuracy is ', val_acc)

      stats['train_accs'].append(train_acc)
      stats['val_accs'].append(val_acc)
      stats['val_accs_ts'].append(t)

      #Alwayse save models
      if True:
        stats['best_val_acc'] = val_acc
        stats['model_t'] = t
        best_pg_state = get_state(program_generator)
        best_ee_state = get_state(execution_engine)
        best_baseline_state = get_state(baseline_model)

      checkpoint = {
        'args': args.__dict__,
        'program_generator_kwargs': pg_kwargs,
        'program_generator_state': best_pg_state,
        'execution_engine_kwargs': ee_kwargs,
        'execution_engine_state': best_ee_state,
        'baseline_kwargs': baseline_kwargs,
        'baseline_state': best_baseline_state,
        'baseline_type': baseline_type,
        'vocab': vocab
      }
      for k, v in stats.items():
        checkpoint[k] = v
      print('Saving checkpoint to %s' % args.checkpoint_path + '_' + str(t))
      torch.save(checkpoint, args.checkpoint_path + '_' + str(t))

      #-------------------------------------------Keze's modification begin-----------------------------------
        if not os.path.exists(os.path.join( get_outputs_path(), "checkpoints")):
          os.mkdir(os.path.join( get_outputs_path(), "checkpoints"))

        torch.save(checkpoint, os.path.join( get_outputs_path(), "checkpoints", str(t)))
      #-------------------------------------------Keze's modification end----------------------------------


      del checkpoint['program_generator_state']
      del checkpoint['execution_engine_state']
      del checkpoint['baseline_state']

      # if t == args.num_iterations:
      #   break


def parse_int_list(s):
  return tuple(int(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = utils.load_program_generator(args.program_generator_start_from)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['refexp_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['refexp_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['refexp_token_to_idx'])
  else:
    kwargs = {
      'encoder_vocab_size': len(vocab['refexp_token_to_idx']),
      'decoder_vocab_size': len(vocab['program_token_to_idx']),
      'wordvec_dim': args.rnn_wordvec_dim,
      'hidden_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
    }
    pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs

def get_execution_engine(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    print('[restart] from', args.execution_engine_start_from)
    ee, kwargs = utils.load_execution_engine(args.execution_engine_start_from)
    # TODO: Adjust vocab?
  else:
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_num_layers': args.module_stem_num_layers,
      'module_dim': args.module_dim,
      'module_residual': args.module_residual == 1,
      'module_batchnorm': args.module_batchnorm == 1,
      'classifier_proj_dim': args.classifier_proj_dim,
      'classifier_downsample': args.classifier_downsample,
      'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
      'classifier_batchnorm': args.classifier_batchnorm == 1,
      'classifier_dropout': args.classifier_dropout,
    }
    ee = ModuleNet(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs


def get_baseline_model(args):
  vocab = utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = utils.load_baseline(args.baseline_start_from)
  elif args.model_type == 'LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'cnn_num_res_blocks': args.cnn_num_res_blocks,
      'cnn_res_block_dim': args.cnn_res_block_dim,
      'cnn_proj_dim': args.cnn_proj_dim,
      'cnn_pooling': args.cnn_pooling,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'stacked_attn_dim': args.stacked_attn_dim,
      'num_stacked_attn': args.num_stacked_attn,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmSaModel(**kwargs)
  if model.rnn.token_to_idx != vocab['refexp_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['refexp_token_to_idx']
      assert vocab['refexp_token_to_idx'][k] == v
    for token, idx in vocab['refexp_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['refexp_token_to_idx'])
  model.cuda()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
  set_mode('eval', [program_generator, execution_engine, baseline_model])
  num_correct, num_samples = 0, 0
  for batch in loader:
    if num_samples % 30 == 0:
      print('process', num_samples, end='\r')
    refexps, _, feats, answers, programs, _, _ = batch

    refexps_var = Variable(refexps.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    answers_var = Variable(feats.cuda(), volatile=True)
    if programs[0] is not None:
      programs_var = Variable(programs.cuda(), volatile=True)

    scores = None # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = utils.load_vocab(args.vocab_json)
      for i in range(refexps.size(0)):
        program_pred = program_generator.sample(Variable(refexps[i:i+1].cuda(), volatile=True))
        program_pred_str = iep.preprocess.decode(program_pred, vocab['program_idx_to_token'])
        program_str = iep.preprocess.decode(programs[i], vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type == 'EE':
        scores = execution_engine(feats_var, programs_var, refexps_var)
        scores = None
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
                          refexps_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred, refexps_var)
    elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(refexps_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if num_samples >= args.num_val_samples:
      break

  set_mode('train', [program_generator, execution_engine, baseline_model])
  acc = float(num_correct) / (num_samples+0.000001)
  return acc


if __name__ == '__main__':

  #-------------------------------------------Keze's modification begin----------------------------------
  nasroot_dir = get_data_paths()['data-pool']
  data_dir = os.path.join( nasroot_dir, 'keze_data' ) 

  src = 'keze_data/.vector_cache'
  dst = './'
  src = os.path.join(get_data_paths()['data-pool'], src)
  dst = os.path.join(os.getcwd(), dst)
  trans_data(src, dst)  


  args = parser.parse_args()

  #tiny config
  args.program_generator_start_from = data_dir + "/backup_models/baseline_18k_single_object_largeBatchSize/program_generator.pt_32000"
  args.train_execution_engine = 1
  args.train_program_generator = 0
  args.model_type = "PG+EE"
  args.num_iterations = 500000
  args.learning_rate = 1e-4
  args.checkpoint_path = data_dir + "/run_fixedPG+EE_ref_singleObject_small/execution_engine_with_CL_extension_for_without_NTM_EntireQuestion_CL_5_epochs.pt"
  args.checkpoint_every_epoch = 1
  args.train_refexp_h5 = data_dir +  "/small_dataset/train_refexps.h5"
  args.train_features_h5 = data_dir +  "/train_features.h5"
  args.val_refexp_h5 = data_dir +  "/small_dataset/val_refexps.h5"
  args.val_features_h5 = data_dir +  "/val_features.h5"
  args.vocab_json = data_dir +  "/vocab.json"

  #-------------------------------------------Keze's modification end----------------------------------

  
  args.batch_size = 8
  args.feature_dim="1024,20,20"

  main(args)
  ###
  ###Results####

  # mar 16 2020: after adding lang attention 
  # training accuracies
  # now IoU = 0.71963, cumulative IoU = 0.545952
  # testing accuracies  on full val_singleobject set
  # 26.67%
  #=============
  # mar 16 2020: without adding lang attention 
  # training accuracies
  # now IoU = 92.9, cumulative IoU = 82.06
  # testing accuracies on full val_singleobject set
  # 42.2%
  #==============



