from typing import Optional, Type
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial, cache
import utils
from flax import linen as nn
import transformer_flax as transformer
from transformer_flax import translate_config
from einshape import jax_einshape as einshape
from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from models_utils import hypercube, build_diag_block, build_bool_sequence, plot_model_consts
import torch



# this is for torch model, avoid using any fancy data structures
@dataclass
class InputData:
  input_id: None
  embedding_mask: None
  demo_cond_k: None
  demo_cond_v: None
  demo_cond_mask: None
  demo_qoi_k: None
  demo_qoi_v: None
  demo_qoi_mask: None
  quest_cond_k: None
  quest_cond_v: None
  quest_cond_mask: None
  quest_qoi_k: None
  quest_qoi_mask: None


def inputdata_transform(func, data):
  '''
  apply func to each dataclass field
  '''
  return InputData(
    input_id = func(data.input_id),
    embedding_mask = func(data.embedding_mask),
    demo_cond_k = func(data.demo_cond_k),
    demo_cond_v = func(data.demo_cond_v),
    demo_cond_mask = func(data.demo_cond_mask),
    demo_qoi_k = func(data.demo_qoi_k),
    demo_qoi_v = func(data.demo_qoi_v),
    demo_qoi_mask = func(data.demo_qoi_mask),
    quest_cond_k = func(data.quest_cond_k),
    quest_cond_v = func(data.quest_cond_v),
    quest_cond_mask = func(data.quest_cond_mask),
    quest_qoi_k = func(data.quest_qoi_k),
    quest_qoi_mask = func(data.quest_qoi_mask),
  )
  


def build_basic_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list, compact):
  '''
  @param:
    cond_len_list: list, lengths of conditions
    qoi_len_list: list, lengths of queries of interest
  @return:
    mask: [sum of (cond_len+qoi_len for each pair), sum of (cond_len+qoi_len for each pair)]
  '''
  assert len(cond_len_list) == len(qoi_kv_len_list) == len(qoi_k_len_list), "Length of lists should be equal"
  num = len(cond_len_list)

  mask_size = sum([cond_len_list[i] + qoi_kv_len_list[i] + qoi_k_len_list[i] for i in range(num)])
  mask = np.zeros((mask_size, mask_size), dtype = bool)

  for i in range(num):
    for j in range(i+1):
      cond_len_i = cond_len_list[i]
      qoi_kv_len_i = qoi_kv_len_list[i]
      qoi_k_len_i = qoi_k_len_list[i]
      cursor_i = sum([cond_len_list[k] + qoi_kv_len_list[k] + qoi_k_len_list[k] for k in range(i)])
      block_size_i = cond_len_i+qoi_kv_len_i+qoi_k_len_i

      cond_len_j = cond_len_list[j]
      qoi_kv_len_j = qoi_kv_len_list[j]
      qoi_k_len_j = qoi_k_len_list[j]
      cursor_j = sum([cond_len_list[k] + qoi_kv_len_list[k] + qoi_k_len_list[k] for k in range(j)])
      block_size_j = cond_len_j+qoi_kv_len_j+qoi_k_len_j

      if i == j:
        mask[cursor_i:cursor_i+block_size_i, cursor_j:cursor_j+block_size_j] = build_diag_block(cond_len_i, qoi_kv_len_i, qoi_k_len_i, compact)
      else:
        if compact:
          mask[cursor_i:cursor_i+block_size_i, cursor_j:cursor_j+cond_len_j+qoi_kv_len_j] = True
        else:   
          mask[cursor_i:cursor_i+block_size_i, cursor_j:cursor_j+block_size_j] = True

  return torch.tensor(mask, dtype=torch.bool)


def build_index_integer(cond_len_list, qoi_kv_len_list, qoi_k_len_list):
  index = []
  num = len(cond_len_list)
  for i in range(num):
    cond_len = cond_len_list[i]
    qoi_kv_len = qoi_kv_len_list[i]
    qoi_k_len = qoi_k_len_list[i]
    index += [i*3] * cond_len + [i*3+1] * qoi_kv_len + [i*3+2] * qoi_k_len
  return torch.tensor(index, dtype=torch.int32)


def build_out_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list, num_range):
  '''
  the output mask, to mask the output of the transformer.
  only the qoi_k are allowed to be outputed, i.e. 000,000,111,000 ...
  @param:
      cond_len_list: list, lengths of conditions
      qoi_kv_len_list: list, lengths of queries of interest for key and value
      qoi_k_len_list: list, lengths of queries of interest for key only
      num_range: tuple, (begin, end), the range of demos to be outputed, if None, all demos except the first one will be outputed
  @return:
      out_mask: [sum of (cond_len+qoi_kv_len+qoi_k_len for each pair)]
  '''
  assert len(cond_len_list) == len(qoi_kv_len_list) == len(qoi_k_len_list), "Length of lists should be equal"
  num = len(cond_len_list)

  out_mask_size = sum([cond_len_list[i] + qoi_kv_len_list[i] + qoi_k_len_list[i] for i in range(num)])
  out_mask = np.zeros((out_mask_size), dtype=bool)
  
  begin, end = num_range

  cursor = 0
  for i in range(num):
    cond_len = cond_len_list[i]
    qoi_kv_len = qoi_kv_len_list[i]
    qoi_k_len = qoi_k_len_list[i]
    pair_size = cond_len + qoi_kv_len + qoi_k_len
    if i >= begin and i < end:
      out_mask[cursor + cond_len + qoi_kv_len: cursor + pair_size] = 1
    cursor += pair_size
  return torch.tensor(out_mask, dtype=torch.bool)


def build_data_sequence_batch(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list, data_quest_qoi_v=None):
  '''
  concatenate the data into a sequence, to be fed into the transformer
  caption not included
  assume leading batch dimension
  '''
  # Handle demo data
  demo_cond = torch.cat([data.demo_cond_k, data.demo_cond_v], dim=-1)  # [batch_size, demo_num, demo_cond_len, k_dim + v_dim]
  demo_qoi_kv = torch.cat([data.demo_qoi_k, data.demo_qoi_v], dim=-1)  # [batch_size, demo_num, demo_qoi_len, k_dim + v_dim]
  demo_qoi_k = torch.cat([data.demo_qoi_k, torch.zeros_like(data.demo_qoi_v)], dim=-1)  # [batch_size, demo_num, demo_qoi_len, k_dim + v_dim]

  # Handle quest data
  if data_quest_qoi_v is None:
    data_quest_qoi_v = torch.zeros_like(data.quest_qoi_k)[..., :data.demo_qoi_v.shape[-1]] # [batch_size, quest_num, quest_qoi_len, v_dim]

  quest_cond = torch.cat([data.quest_cond_k, data.quest_cond_v], dim=-1)  # [batch_size, quest_num, quest_cond_len, k_dim + v_dim]
  quest_qoi_kv = torch.cat([data.quest_qoi_k, data_quest_qoi_v], dim=-1)  # [batch_size, quest_num, quest_qoi_len, k_dim + v_dim]
  quest_qoi_k = torch.cat([data.quest_qoi_k, torch.zeros_like(data_quest_qoi_v)], dim=-1)  # [batch_size, quest_num, quest_qoi_len, k_dim + v_dim]
  
  demo_num = len(cond_bool_list)-1

  sequence = []
  for i in range(demo_num):
    if cond_bool_list[i]: sequence.append(demo_cond[:,i,...])
    if qoi_kv_bool_list[i]: sequence.append(demo_qoi_kv[:,i,...])
    if qoi_k_bool_list[i]: sequence.append(demo_qoi_k[:,i,...])
  if cond_bool_list[-1]: sequence.append(quest_cond[:,0,...])
  if qoi_kv_bool_list[-1]: sequence.append(quest_qoi_kv[:,0,...])
  if qoi_k_bool_list[-1]: sequence.append(quest_qoi_k[:,0,...])

  sequence = torch.cat(sequence, dim = -2) # [batch_size, seq_length, dim]
  return sequence

def build_data_mask_batch(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list):
  '''
  concatenate the masks in data, to be fed into the transformer
  caption mask excluded
  assume leading batch dimension
  '''
  demo_num = len(cond_bool_list)-1

  mask = []

  for i in range(demo_num):
    if cond_bool_list[i]: mask.append(data.demo_cond_mask[:,i,...])
    if qoi_kv_bool_list[i]: mask.append(data.demo_qoi_mask[:,i,...])
    if qoi_k_bool_list[i]: mask.append(data.demo_qoi_mask[:,i,...])
  if cond_bool_list[-1]: mask.append(data.quest_cond_mask[:,0,...])
  if qoi_kv_bool_list[-1]: mask.append(data.quest_qoi_mask[:,0,...])
  if qoi_k_bool_list[-1]: mask.append(data.quest_qoi_mask[:,0,...])

  mask = torch.cat(mask, dim = -1) # [batch_size, seq_length]
  return mask

