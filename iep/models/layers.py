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

    self.lstm1 = nn.LSTM(tq_dim, in_dim, bidirectional=True, batch_first=True)
    self.lstm2 = nn.GRU(in_dim*2, int(in_dim/2), bidirectional=True, batch_first=True)

    self.lstm11 = nn.LSTM(q_dim, in_dim, bidirectional=True, batch_first=True)
    self.lstm22 = nn.GRU(in_dim*2, int(in_dim/2), bidirectional=True, batch_first=True)

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
    q = q.unsqueeze(0).cuda()
    t_q = t_q.unsqueeze(0).cuda() # to make t_q as 1Xseq_lenX300

    tq_lstm, _ = self.lstm1(t_q)
    tq_lstm, _ = self.lstm2(tq_lstm) # output size of h_lstm is batchXseq_lenX300 ... so we need to

    q_lstm, _ = self.lstm11(q)
    q_lstm, _ = self.lstm22(q_lstm) # output size of h_lstm is batchXseq_lenX300 ... so we need to

    #txt_conv = self.conv11( F.relu(self.conv1(x) * self.lang1(q).view(q.shape[0],-1,1,1)) ) * h_lstm[:,-1].view(t_q.shape[0],-1,1,1)
    txt_conv = self.conv11( F.relu(self.conv1(x) * q_lstm[:,-1].view(q.shape[0],-1,1,1)) ) * tq_lstm[:,-1].view(t_q.shape[0],-1,1,1) 

    if self.with_batchnorm:
      out = F.relu(self.bn1(txt_conv))
      out = self.bn2(self.conv2(out))
    else:
      out = self.conv2(F.relu(txt_conv))
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
