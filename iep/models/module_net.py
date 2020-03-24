#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models
import numpy as np

from iep.models.layers import ResidualBlock, ResidualBlock_LangAttention, GlobalAveragePool, Flatten
import iep.programs
import torchtext
import re


class ConcatBlock(nn.Module):
  def __init__(self, dim, with_residual=True, with_batchnorm=True):
    super(ConcatBlock, self).__init__()
    self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)
    self.res_block = ResidualBlock(dim, with_residual=with_residual,
                        with_batchnorm=with_batchnorm)

  def forward(self, x, y):
    out = torch.cat([x, y], 1) # Concatentate along depth
    out = F.relu(self.proj(out))
    out = self.res_block(out)
    return out


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True):
  layers = []
  prev_dim = feature_dim
  for i in range(num_layers):
    layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=3, padding=1))
    if with_batchnorm:
      layers.append(nn.BatchNorm2d(module_dim))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = module_dim
  return nn.Sequential(*layers)


def build_classifier(module_C, module_H, module_W, 
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=0):

  res_block = ResidualBlock(module_C,
              with_residual=True,
              with_batchnorm=False)
  layers = [res_block]

  layers.append(nn.Conv2d(module_C, module_C, kernel_size=1))
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(module_dim))
  layers.append(nn.ReLU(inplace=True))

  upsample = nn.Upsample(size=[320,320],mode='bilinear')

  layers.append(upsample)
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(module_dim))
  layers.append(nn.ReLU(inplace=True))

  layers.append(nn.Conv2d(module_C, module_C, kernel_size=1))
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(module_dim))
  layers.append(nn.ReLU(inplace=True))
 
  layers.append(nn.Conv2d(module_C, module_C//4, kernel_size=1))
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(module_dim))
  layers.append(nn.ReLU(inplace=True))

  layers.append(nn.Conv2d(module_C//4, 4, kernel_size=1))
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(module_dim))
  layers.append(nn.ReLU(inplace=True))

  layers.append(nn.Conv2d(4, 2, kernel_size=1))
  if with_batchnorm:
    layers.append(nn.BatchNorm2d(module_dim))

  return nn.Sequential(*layers)


class ModuleNet(nn.Module):
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               module_dim=128,
               module_residual=True,
               module_batchnorm=False,
               classifier_proj_dim=512,
               classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               verbose=True):
    super(ModuleNet, self).__init__()

    self.stem = build_stem(feature_dim[0], module_dim,
                           num_layers=stem_num_layers,
                           with_batchnorm=stem_batchnorm)
    if verbose:
      print('Here is my stem:')
      print(self.stem)
    self.glove = torchtext.vocab.GloVe(name="840B", dim=300)   # embedding size = 300
    
    module_H, module_W = feature_dim[1], feature_dim[2]
    self.classifier = build_classifier(module_dim, module_H, module_W, 
                                       classifier_fc_layers,
                                       classifier_proj_dim,
                                       classifier_downsample,
                                       with_batchnorm=classifier_batchnorm,
                                       dropout=classifier_dropout
                                       )
    if verbose:
      print('Here is my classifier:')
      print(self.classifier)
    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False

    self.function_modules = {}
    self.function_modules_num_inputs = {}
    self.vocab = vocab

    
    print("vocab['program_token_to_idx']={}".format(vocab['program_token_to_idx']))
    for fn_str in vocab['program_token_to_idx']:
      fn_str = str(fn_str)

      #FIXME: me
      if (fn_str.split("[")[0] in self.function_modules):
        continue;
      
      num_inputs = iep.programs.get_num_inputs(fn_str)
      self.function_modules_num_inputs[fn_str.split("[")[0]] = num_inputs
      #FIXME:
      # if fn_str == 'scene' or num_inputs == 1:
      #   mod = ResidualBlock(module_dim,
      #           with_residual=module_residual,
      #           with_batchnorm=module_batchnorm)
      # elif num_inputs == 2:
      #   mod = ConcatBlock(module_dim,
      #           with_residual=module_residual,
      #           with_batchnorm=module_batchnorm)
      if fn_str == 'scene':
        mod = ResidualBlock(module_dim,
                with_residual=module_residual,
                with_batchnorm=module_batchnorm)
      elif num_inputs == 1 and len(fn_str.split("[")) < 2:
        mod = ResidualBlock(module_dim,
                with_residual=module_residual,
                with_batchnorm=module_batchnorm)
      elif num_inputs == 1 and len(fn_str.split("[")) >= 2:
        mod = ResidualBlock_LangAttention(module_dim,
                with_residual=module_residual,
                with_batchnorm=module_batchnorm)
      elif num_inputs == 2:
        mod = ConcatBlock(module_dim,
                with_residual=module_residual,
                with_batchnorm=module_batchnorm)

      #FIXME: me
      #self.add_module(fn_str, mod)
      #self.function_modules[fn_str] = mod
      self.add_module(fn_str.split("[")[0], mod)
      self.function_modules[fn_str.split("[")[0]] = mod

    self.save_module_outputs = False


  def expand_answer_vocab(self, answer_to_idx, std=0.01, init_b=-50):
    # TODO: This is really gross, dipping into private internals of Sequential
    final_linear_key = str(len(self.classifier._modules) - 1)
    final_linear = self.classifier._modules[final_linear_key]

    old_weight = final_linear.weight.data
    old_bias = final_linear.bias.data
    old_N, D = old_weight.size()
    new_N = 1 + max(answer_to_idx.values())
    new_weight = old_weight.new(new_N, D).normal_().mul_(std)
    new_bias = old_bias.new(new_N).fill_(init_b)
    new_weight[:old_N].copy_(old_weight)
    new_bias[:old_N].copy_(old_bias)

    final_linear.weight.data = new_weight
    final_linear.bias.data = new_bias


  def _forward_modules_json(self, feats, program,refs):
    def gen_hook(i, j):
      def hook(grad):
        self.all_module_grad_outputs[i][j] = grad.data.cpu().clone()
      return hook

    self.all_module_outputs = []
    self.all_module_grad_outputs = []
    # We can't easily handle minibatching of modules, so just do a loop
    N = feats.size(0)
    final_module_outputs = []
    for i in range(N):
      if self.save_module_outputs:
        self.all_module_outputs.append([])
        self.all_module_grad_outputs.append([None] * len(program[i]))
      module_outputs = []
      for j, f in enumerate(program[i]):
        f_str = iep.programs.function_to_str(f)
        #FIXME:
        #module = self.function_modules[f_str]
        module = self.function_modules[f_str.split("[")[0]]
        if f_str == 'scene':
          module_inputs = [feats[i:i+1]]
        else:
          module_inputs = [module_outputs[j] for j in f['inputs']]
        module_outputs.append(module(*module_inputs))
        if self.save_module_outputs:
          self.all_module_outputs[-1].append(module_outputs[-1].data.cpu().clone())
          module_outputs[-1].register_hook(gen_hook(i, j))
      final_module_outputs.append(module_outputs[-1])
    final_module_outputs = torch.cat(final_module_outputs, 0)
    return final_module_outputs


  def _forward_modules_ints_helper(self, feats, program, i, j,refs):
    used_fn_j = True
    if j < program.size(1):
      fn_idx = program.data[i, j]
      fn_idx = int(fn_idx.cpu().numpy())
      fn_str = self.vocab['program_idx_to_token'][fn_idx]
    else:
      used_fn_j = False
      fn_str = 'scene'
    if fn_str == '<NULL>':
      used_fn_j = False
      fn_str = 'scene'
    elif fn_str == '<START>':
      used_fn_j = False
      return self._forward_modules_ints_helper(feats, program, i, j + 1, refs)
    if used_fn_j:
      self.used_fns[i, j] = 1
    j += 1
    #FIXME:
    #module = self.function_modules[fn_str]
    module = self.function_modules[fn_str.split("[")[0]]
    if fn_str == 'scene':
      module_inputs = [feats[i:i+1]]
    else:
      num_inputs = self.function_modules_num_inputs[fn_str.split("[")[0]]
      module_inputs = []
      
      while len(module_inputs) < num_inputs:
        cur_input, j = self._forward_modules_ints_helper(feats, program, i, j, refs)
        module_inputs.append(cur_input)

      if(len(fn_str.split("[")) >= 2):
        lang_txt_inp = fn_str.split("[")[1][:-1]
        lang_txt_inp_list = re.split('[, ;\'"?\.!()]',lang_txt_inp.strip())
        txt_vec_list = []
        for l_i in range(0, len(lang_txt_inp_list)):
          wrd = lang_txt_inp_list[l_i].strip()
          if(len(wrd) == 0):
            continue
          txt_vec_list.append(torch.Tensor(self.glove[wrd]).view(1,-1))

        for i_i in range(0,len(txt_vec_list)):
          if(i_i == 0):
            txt_inp = txt_vec_list[i_i]
          else:
            txt_inp = torch.cat((txt_inp, txt_vec_list[i_i]),0)
        
        if(len(txt_vec_list) == 0):
          module_inputs.append(torch.Tensor(self.glove['NULL']).view(1,-1))
        else:  
          module_inputs.append(txt_inp)  # shape of entire_txt_inp is seq_lenX300
        
    if(len(fn_str.split("[")) >= 2):
      total_txt = refs[i].data.cpu().numpy()
      total_txt_vec_list = []
      
      for l_i in range(0, len(total_txt)):
        if(total_txt[l_i] == 0):
          break
        wrd = self.vocab['refexp_idx_to_token'][total_txt[l_i]]
        if(len(wrd) == 0):
          continue
        total_txt_vec_list.append(torch.Tensor(self.glove[wrd]).view(1,-1))
      
      for i_i in range(0,len(total_txt_vec_list)):
        if(i_i == 0):
          entire_txt_inp = total_txt_vec_list[i_i]
        else:
          entire_txt_inp = torch.cat((entire_txt_inp, total_txt_vec_list[i_i]),0)

      if(len(total_txt_vec_list) == 0):
        module_inputs.append(torch.Tensor(self.glove['NULL']).view(1,-1))
      else:
        module_inputs.append(entire_txt_inp) # shape of entire_txt_inp is seq_lenX300

    module_output = module(*module_inputs)
    return module_output, j


  def _forward_modules_ints(self, feats, program, refs):
    """
    feats: FloatTensor of shape (N, C, H, W) giving features for each image
    program: LongTensor of shape (N, L) giving a prefix-encoded program for
      each image.
    """
    N = feats.size(0)
    final_module_outputs = []
    self.used_fns = torch.Tensor(program.size()).fill_(0)
    for i in range(N):
      cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0, refs)
      final_module_outputs.append(cur_output)
    self.used_fns = self.used_fns.type_as(program.data).float()
    final_module_outputs = torch.cat(final_module_outputs, 0)
    return final_module_outputs


  def forward(self, x, program, refs):
    N = x.size(0)
    assert N == len(program)

    feats = self.stem(x)

    if type(program) is list or type(program) is tuple:
      final_module_outputs = self._forward_modules_json(feats, program, refs)
    elif type(program) is torch.Tensor and program.dim() == 2:
      final_module_outputs = self._forward_modules_ints(feats, program, refs)
    elif type(program) is Variable and program.dim() == 2:
      final_module_outputs = self._forward_modules_ints(feats, program, refs)
    else:
      raise ValueError('Unrecognized program format')

    # After running modules for each input, concatenat the outputs from the
    # final module and run the classifier.
    batch_size = final_module_outputs.shape[0]
    module_dim = final_module_outputs.shape[1]
    out = self.classifier(final_module_outputs)
    return out
