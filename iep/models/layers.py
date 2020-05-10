#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
    self.with_batchnorm = with_batchnorm
    if with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim)
      self.bn2 = nn.BatchNorm2d(out_dim)
    self.with_residual = with_residual
    if in_dim == out_dim or not with_residual:
      self.proj = None
    else:
      self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

  def forward(self, x):
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out

#FIXME: x: 1X128X20X20, q:1X50, W:50X128 ... Wq-> 1X128 --> 1X128X1X1
class ResidualBlock_LangAttention(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True, q_dim=300, tq_dim = 300):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock_LangAttention, self).__init__()
    #self.lang1 = nn.Linear(q_dim, in_dim)

    self.lstm1 = nn.LSTM(tq_dim, in_dim, bidirectional=False, batch_first=True)
    #self.lstm2 = nn.GRU(in_dim*2, int(in_dim/2), bidirectional=True, batch_first=True)

    self.lstm11 = nn.LSTM(q_dim, in_dim, bidirectional=False, batch_first=True)
    #self.lstm11 = nn.LSTM(q_dim, int(in_dim/2), bidirectional=True, batch_first=True)
    #self.lstm11 = nn.LSTM(q_dim, in_dim, bidirectional=False, batch_first=True)
    #self.lstm22 = nn.GRU(in_dim*2, int(in_dim/2), bidirectional=True, batch_first=True)

    #self.attention_layer_q = Attention(in_dim)
    #self.attention_layer_tq = Attention(in_dim)

    #self.lang_tq = nn.Linear(tq_dim, in_dim)
    self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
    self.conv11 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

    self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
    self.with_batchnorm = with_batchnorm
    if with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim)
      self.bn2 = nn.BatchNorm2d(out_dim)
    self.with_residual = with_residual
    if in_dim == out_dim or not with_residual:
      self.proj = None
    else:
      self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

  #expect t_q to be seq_len X 300, where 300 is word2vec embedding of each word.
  def forward(self, x, q, t_q):
    # q = q.unsqueeze(0).cuda()
    # t_q = t_q.unsqueeze(0).cuda() # to make t_q as 1Xseq_lenX300

    #packed_embedded_q = nn.utils.rnn.pack_padded_sequence(q, [q.shape[1]],batch_first=True)
    #packed_embedded_tq = nn.utils.rnn.pack_padded_sequence(t_q, [t_q.shape[1]],batch_first=True)

    # tq_lstm, _ = self.lstm1(t_q)
    # #tq_lstm, _ = self.lstm2(tq_lstm) # output size of h_lstm is batchXseq_lenX300 ... so we need to

    # q_lstm, _ = self.lstm11(q)
    # #q_lstm, _ = self.lstm22(q_lstm) # output size of h_lstm is batchXseq_lenX300 ... so we need to

    #q_attn = self.attention_layer_q(q_lstm, q.shape[1])
    #tq_attn = self.attention_layer_q(tq_lstm, t_q.shape[1])

    #lang encode without attention
    #txt_conv = self.conv11( F.relu(self.conv1(x) * q_lstm[:,-1].view(q.shape[0],-1,1,1)) ) * tq_lstm[:,-1].view(t_q.shape[0],-1,1,1)
    
    #lang encode with attention
    #txt_conv = self.conv11( F.relu(self.conv1(x) * q_attn.view(q.shape[0],-1,1,1)) ) * tq_attn.view(t_q.shape[0],-1,1,1)
    # txt_conv = self.conv11( F.relu(self.conv1(x) * q_lstm[:,-1].view(q.shape[0],-1,1,1)) ) * tq_lstm[:,-1].view(t_q.shape[0],-1,1,1)

    #txt_conv = (self.conv1(x) * q_lstm[:,-1].view(q.shape[0],-1,1,1) ) * tq_lstm[:,-1].view(t_q.shape[0],-1,1,1)
    #txt_conv = (self.conv1(x) * q_lstm[:,-1].view(q.shape[0],-1,1,1) ) 
    #txt_conv = self.conv1(x)

    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out


class GlobalAveragePool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).mean(2).squeeze(2)


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim=100, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
    def forward(self, x, step_dim, mask=None):
        self.step_dim = step_dim
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + nn.Parameter(torch.zeros(step_dim)).cuda()
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
