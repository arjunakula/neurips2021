#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import iep.programs


def _dataset_to_tensor(dset, mask=None):
  arr = np.asarray(dset, dtype=np.int64)
  if mask is not None:
    arr = arr[mask]
  tensor = torch.LongTensor(arr)
  return tensor


class ClevrDataset(Dataset):
  def __init__(self, refexp_h5, feature_h5, vocab, mode='prefix',
               image_h5=None, max_samples=None, refexp_families=None,
               image_idx_start_from=None):
    mode_choices = ['prefix', 'postfix']
    if mode not in mode_choices:
      raise ValueError('Invalid mode "%s"' % mode)
    self.image_h5 = image_h5
    self.vocab = vocab
    self.feature_h5 = feature_h5
    self.mode = mode
    self.max_samples = max_samples

    mask = None

    refexp_h5 = h5py.File(refexp_h5, 'r')


    if refexp_families is not None:
      # Use only the specified families
      all_families = np.asarray(refexp_h5['refexp_families'])
      N = all_families.shape[0]
      print(refexp_families)
      target_families = np.asarray(refexp_families)[:, None]
      mask = (all_families == target_families).any(axis=0)
    if image_idx_start_from is not None:
      all_image_idxs = np.asarray(refexp_h5['image_idxs'])
      mask = all_image_idxs >= image_idx_start_from

    # Data from the refexp file is small, so read it all into memory
    print('Reading refexp data into memory')
    self.all_refexps = _dataset_to_tensor(refexp_h5['refexps'], mask)
    self.all_image_idxs = _dataset_to_tensor(refexp_h5['image_idxs'], mask)
    self.all_programs = None
    if 'programs' in refexp_h5:
      self.all_programs = _dataset_to_tensor(refexp_h5['programs'], mask)
    #self.all_answers = _dataset_to_tensor(refexp_h5['answers'], mask)
    assert mask == None
    self.all_answers = refexp_h5['answers']


  def __getitem__(self, index):
    refexp = self.all_refexps[index]
    image_idx = self.all_image_idxs[index]
    _tmp = np.asarray(self.all_answers[index], dtype=np.int64)
    answer = torch.LongTensor(_tmp)

    program_seq = None
    if self.all_programs is not None:
      program_seq = self.all_programs[index]

    image = None
    if self.image_h5 is not None:
      image = self.image_h5['images'][image_idx]
      image = torch.FloatTensor(np.asarray(image, dtype=np.float32))

    feats = self.feature_h5['features'][image_idx]
    feats = torch.FloatTensor(np.asarray(feats, dtype=np.float32))

    program_json = None
    if program_seq is not None:
      program_json_seq = []
      for fn_idx in program_seq:
        fn_idx=int(fn_idx.data.cpu().numpy())
        fn_str = self.vocab['program_idx_to_token'][fn_idx]
        if fn_str == '<START>' or fn_str == '<END>': continue
        fn = iep.programs.str_to_function(fn_str)
        program_json_seq.append(fn)
      if self.mode == 'prefix':
        program_json = iep.programs.prefix_to_list(program_json_seq)
      elif self.mode == 'postfix':
        program_json = iep.programs.postfix_to_list(program_json_seq)

    return (refexp, image, feats, answer, program_seq, program_json, image_idx)

  def __len__(self):
    if self.max_samples is None:
      return self.all_refexps.size(0)
    else:
      return min(self.max_samples, self.all_refexps.size(0))


class ClevrDataLoader(DataLoader):
  def __init__(self, **kwargs):
    if 'refexp_h5' not in kwargs:
      raise ValueError('Must give refexp_h5')
    if 'feature_h5' not in kwargs:
      raise ValueError('Must give feature_h5')
    if 'vocab' not in kwargs:
      raise ValueError('Must give vocab')

    feature_h5_path = kwargs.pop('feature_h5')
    print('Reading features from ', feature_h5_path)
    self.feature_h5 = h5py.File(feature_h5_path, 'r')

    self.image_h5 = None
    if 'image_h5' in kwargs:
      image_h5_path = kwargs.pop('image_h5')
      print('Reading images from ', image_h5_path)
      self.image_h5 = h5py.File(image_h5_path, 'r')

    vocab = kwargs.pop('vocab')
    mode = kwargs.pop('mode', 'prefix')

    refexp_families = kwargs.pop('refexp_families', None)
    max_samples = kwargs.pop('max_samples', None)
    refexp_h5_path = kwargs.pop('refexp_h5')
    image_idx_start_from = kwargs.pop('image_idx_start_from', None)
    print('Reading refexps from ', refexp_h5_path)
    self.dataset = ClevrDataset(refexp_h5_path, self.feature_h5, vocab, mode,
                                image_h5=self.image_h5,
                                max_samples=max_samples,
                                refexp_families=refexp_families,
                                image_idx_start_from=image_idx_start_from)
    kwargs['collate_fn'] = clevr_collate
    super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

  def close(self):
    if self.image_h5 is not None:
      self.image_h5.close()
    if self.feature_h5 is not None:
      self.feature_h5.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


def clevr_collate(batch):
  transposed = list(zip(*batch))
  refexp_batch = default_collate(transposed[0])
  image_batch = transposed[1]
  if any(img is not None for img in image_batch):
    image_batch = default_collate(image_batch)
  feat_batch = transposed[2]
  if any(f is not None for f in feat_batch):
    feat_batch = default_collate(feat_batch)
  answer_batch = default_collate(transposed[3])
  program_seq_batch = transposed[4]
  if transposed[4][0] is not None:
    program_seq_batch = default_collate(transposed[4])
  program_struct_batch = transposed[5]
  image_id_batch = default_collate(transposed[6])
  return [refexp_batch, image_batch, feat_batch, answer_batch, program_seq_batch, program_struct_batch, image_id_batch]
