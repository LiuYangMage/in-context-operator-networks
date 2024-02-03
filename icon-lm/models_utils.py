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
import jax.tree_util as tree
import itertools
import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from matplotlib.collections import LineCollection


def hypercube(n):
    return list(itertools.product([0, 1], repeat=n))


def build_diag_block(cond_len, qoi_kv_len, qoi_k_len, compact):
  diag_block = np.zeros((cond_len + qoi_kv_len + qoi_k_len, cond_len + qoi_kv_len + qoi_k_len), dtype = bool)
  diag_block[:, :cond_len] = 1
  if compact:
    diag_block[cond_len:cond_len+qoi_kv_len, cond_len:cond_len+qoi_kv_len] = 1
  else:
    diag_block[cond_len:cond_len+qoi_kv_len, cond_len:] = 1
  diag_block[cond_len+qoi_kv_len:, cond_len+qoi_kv_len:] = np.eye(qoi_k_len, dtype = bool)
  return diag_block


def build_bool_sequence(demo_num, mode, shot_num_min = None):

  if mode == 'full':
    cond_list = [True] * demo_num + [True]
    qoi_kv_list = [True] * demo_num + [True]
    qoi_k_list = [True] * demo_num + [True]
  elif mode == 'no_query':
    cond_list = [True] * demo_num + [True]
    qoi_kv_list = [True] * demo_num + [True]
    qoi_k_list = [False] * demo_num + [False]
  elif mode == 'train':
    # this is for training, no qoi_kv for the last pair, no qoi_k for the first shot_num_min pairs
    cond_list = [True] * demo_num + [True]
    qoi_kv_list = [True] * demo_num + [False] # no qoi_kv for quest
    qoi_k_list = [True if i >= shot_num_min else False for i in range(demo_num)] + [True] # no qoi_k if less than shot_num_min shots
  elif mode == 'test':
    # this is for testing, only see qoi_k for the las pair, i.e. the quest
    cond_list = [True] * demo_num + [True]
    qoi_kv_list = [True] * demo_num + [False] # no qoi_kv for quest
    qoi_k_list = [False] * demo_num + [True] # only see qoi_k for quest
  else:
    raise ValueError('not supported mode: {}'.format(mode))
  return cond_list, qoi_kv_list, qoi_k_list


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

  return jnp.array(mask)



def build_index_integer(cond_len_list, qoi_kv_len_list, qoi_k_len_list):
  index = []
  num = len(cond_len_list)
  for i in range(num):
    cond_len = cond_len_list[i]
    qoi_kv_len = qoi_kv_len_list[i]
    qoi_k_len = qoi_k_len_list[i]
    index += [i*3] * cond_len + [i*3+1] * qoi_kv_len + [i*3+2] * qoi_k_len
  return jnp.array(index)

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
  return jnp.array(out_mask, dtype=bool)



def build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list, data_quest_qoi_v = None):
  '''
  concatenate the data into a sequence, to be fed into the transformer
  caption not included, since we need to add index matrix later
  '''
  # Handle demo data
  demo_cond = jnp.concatenate([data.demo_cond_k, data.demo_cond_v], axis = -1) # [demo_num, demo_cond_len, k_dim + v_dim]
  demo_qoi_kv = jnp.concatenate([data.demo_qoi_k, data.demo_qoi_v], axis = -1) # [demo_num, demo_qoi_len, k_dim + v_dim]
  demo_qoi_k = jnp.pad(data.demo_qoi_k, ((0,0),(0,0),(0, data.demo_qoi_v.shape[-1])), mode='constant', constant_values=0) # [demo_num, demo_qoi_len, k_dim + v_dim]

  # Handle quest data
  if data_quest_qoi_v is None:
    # [quest_num, quest_qoi_len, v_dim]
    data_quest_qoi_v = jnp.zeros((data.quest_qoi_k.shape[0], data.quest_qoi_k.shape[1], data.demo_qoi_v.shape[-1]))
  quest_cond = jnp.concatenate([data.quest_cond_k, data.quest_cond_v], axis = -1) # [quest_num, quest_cond_len, k_dim + v_dim]
  quest_qoi_kv = jnp.concatenate([data.quest_qoi_k, data_quest_qoi_v], axis = -1) # [quest_num, quest_qoi_len, k_dim + v_dim]
  quest_qoi_k = jnp.pad(data.quest_qoi_k, ((0,0),(0,0),(0, data_quest_qoi_v.shape[-1])), mode='constant', constant_values=0) # [quest_num, quest_qoi_len, k_dim + v_dim]
  
  demo_num = len(data.demo_cond_k)

  sequence = []
  for i in range(demo_num):
    if cond_bool_list[i]: sequence.append(demo_cond[i])
    if qoi_kv_bool_list[i]: sequence.append(demo_qoi_kv[i])
    if qoi_k_bool_list[i]: sequence.append(demo_qoi_k[i])
  if cond_bool_list[-1]: sequence.append(quest_cond[0])
  if qoi_kv_bool_list[-1]: sequence.append(quest_qoi_kv[0])
  if qoi_k_bool_list[-1]: sequence.append(quest_qoi_k[0])
  sequence = jnp.concatenate(sequence, axis = 0) # [sequence length, k_dim + v_dim]

  return sequence

def build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list):
  '''
  concatenate the masks in data, to be fed into the transformer
  caption mask excluded
  '''
  demo_num = len(data.demo_cond_k)

  mask = []

  for i in range(demo_num):
    if cond_bool_list[i]: mask.append(data.demo_cond_mask[i])
    if qoi_kv_bool_list[i]: mask.append(data.demo_qoi_mask[i])
    if qoi_k_bool_list[i]: mask.append(data.demo_qoi_mask[i])
  if cond_bool_list[-1]: mask.append(data.quest_cond_mask[0])
  if qoi_kv_bool_list[-1]: mask.append(data.quest_qoi_mask[0])
  if qoi_k_bool_list[-1]: mask.append(data.quest_qoi_mask[0])

  mask = jnp.concatenate(mask, axis = 0) # [sequence length]
  return mask




def plot_model_consts(basic_mask = None, index_matrix = None, out_mask = None):
  fig = plt.figure(figsize = (10,10))
  if basic_mask is not None:
    plt.subplot(2,2,1)
    plt.imshow(basic_mask, interpolation = "nearest")
    plt.subplot(2,2,2)
    plt.imshow((basic_mask @ basic_mask).astype(bool), interpolation = "nearest")
    assert jnp.allclose((basic_mask @ basic_mask).astype(bool), basic_mask.astype(bool))
  if index_matrix is not None:
    plt.subplot(2,2,3)
    plt.imshow(index_matrix.T, aspect= (index_matrix.shape[0]/index_matrix.shape[1]), interpolation = "nearest")
  if out_mask is not None:
    plt.subplot(2,2,4)
    plt.imshow(out_mask[None, :], aspect= out_mask.shape[0], interpolation = "nearest")
  return fig
