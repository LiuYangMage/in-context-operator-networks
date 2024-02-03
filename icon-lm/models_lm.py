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

from models_utils import *


def build_matrices_from_data_shape(data_shape, config, compact, mode, caption_len, shot_num_min, return_shape_list = False):
  '''
  data_shape is the shape of data, usually obtained by tree.tree_map(lambda x: x.shape, data)
  '''
  demo_num = data_shape.demo_cond_k[0]
  demo_cond_len = data_shape.demo_cond_k[1]
  demo_qoi_len = data_shape.demo_qoi_k[1]
  quest_cond_len = data_shape.quest_cond_k[1]
  quest_qoi_len = data_shape.quest_qoi_k[1]

  cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = build_bool_sequence(demo_num, mode, shot_num_min)
  cond_len_list_raw = [demo_cond_len] * demo_num + [quest_cond_len]
  qoi_kv_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
  qoi_k_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
  cond_len_list = [i * j for i, j in zip(cond_bool_list, cond_len_list_raw)]
  qoi_kv_len_list = [i * j for i, j in zip(qoi_kv_bool_list, qoi_kv_len_list_raw)]
  qoi_k_len_list = [i * j for i, j in zip(qoi_k_bool_list, qoi_k_len_list_raw)]
  
  basic_mask = build_basic_mask(cond_len_list = cond_len_list, 
                                qoi_kv_len_list = qoi_kv_len_list, 
                                qoi_k_len_list = qoi_k_len_list, 
                                compact = compact)
  if config['index_mode'] == 'learn':
    index_matrix = build_index_integer(cond_len_list= cond_len_list,
                                      qoi_kv_len_list = qoi_kv_len_list,
                                      qoi_k_len_list = qoi_k_len_list)
  else:
    raise ValueError('not supported index mode: {}'.format(config['index_mode']))
  
  out_mask = build_out_mask(cond_len_list = cond_len_list,
                          qoi_kv_len_list= qoi_kv_len_list,
                          qoi_k_len_list = qoi_k_len_list,
                          num_range = (shot_num_min, demo_num + 1))
  # add prefix
  if caption_len > 0:
    basic_mask = jnp.pad(basic_mask, ((caption_len, 0), (0, 0)), mode='constant', constant_values=0)
    basic_mask = jnp.pad(basic_mask, ((0, 0), (caption_len, 0)), mode='constant', constant_values=1)
    out_mask = jnp.pad(out_mask, (caption_len, 0), mode='constant', constant_values=0)
  
  if return_shape_list:
    return basic_mask, index_matrix, out_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list
  else:
    return basic_mask, index_matrix, out_mask


class IconGPTModel(nn.Module):
  '''
  IconGPT model
  '''
  config: dict
  # these matrices are constant, so we can pre-compute them. They are used in the __call__ function 

  # x and x' may differ, due to shot_num_min
  basic_mask_without_caption: jnp.ndarray # [x, x]
  index_matrix_without_caption: jnp.ndarray # [x, index_dim]
  out_mask_without_caption: jnp.ndarray # [x]
  
  basic_mask_with_caption: jnp.ndarray # [x' + caption_len, x' + caption_len]
  index_matrix_with_caption: jnp.ndarray # [x', index_dim]
  out_mask_with_caption: jnp.ndarray # [x' + caption_len]

  def setup(self):

    self.pre_projection = nn.Dense(self.config['transformer']['model_dim'], name="pre_projection")

    if self.config['index_mode'] == 'learn':
    # trainable positional embedding
      self.func_pos_embedding = nn.Embed((self.config['demo_max_num']) * 3, self.config['transformer']['model_dim'], name="func_pos_embedding")

    if self.config['caption_len'] > 0:
      self.caption_projection = nn.Dense(self.config['transformer']['model_dim'], name="caption_projection")
      self.positional_embedding = utils.get_positional_encoding(self.config['caption_len'], self.config['transformer']['model_dim'])
    self.transformer = transformer.SelfAttnTransformer(**translate_config(self.config['transformer']), name='transformer')
    self.post_projection = nn.Dense(self.config['out_dim'], name="post_projection")



  def basic_forward(self, data, mode, index_matrix, basic_mask, caption_len, shot_num_min):
    '''
    @param:
      data, named tuple, with the following field:
        demo_cond_k: [demo_num, demo_cond_len, cond_k_dim]
        demo_cond_v: [demo_num, demo_cond_len, cond_v_dim]
        demo_cond_mask: [demo_num, demo_cond_len]
        demo_qoi_k: [demo_num, demo_qoi_len, qoi_k_dim]
        demo_qoi_v: [demo_num, demo_qoi_len, qoi_v_dim]
        demo_qoi_mask: [demo_num, demo_qoi_len]
        quest_cond_k: [1, quest_cond_len, cond_k_dim]
        quest_cond_v: [1, quest_cond_len, cond_v_dim]
        quest_cond_mask: [1, quest_cond_len]
        quest_qoi_k: [1, quest_qoi_len, qoi_k_dim]
        quest_qoi_v: [1, quest_qoi_len, qoi_v_dim] # this is the ground truth, should never be used here!
        quest_qoi_mask: [1, quest_qoi_len] # this is also useless
    '''
    demo_num = len(data.demo_cond_k)
    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = build_bool_sequence(demo_num, mode = mode, shot_num_min = shot_num_min)
    sequence = build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
    
    if self.config['index_mode'] == 'learn':
      sequence = self.pre_projection(sequence)
      sequence = sequence + self.func_pos_embedding(index_matrix)
    else:
      raise ValueError('not supported index mode: {}'.format(self.config['index_mode']))
    
    if caption_len > 0:
      if self.config['caption_feature'] == 'embedding':
        projected_caption = self.caption_projection(data.embedding_raw) # [caption_len, model_dim]
      elif self.config['caption_feature'] == 'input_id':
        one_hot_caption = jax.nn.one_hot(data.input_id, num_classes = self.config['caption_vocab_size']) # [caption_len, vocab_size]
        projected_caption = self.caption_projection(one_hot_caption) # [caption_len, model_dim]
        projected_caption = projected_caption + self.positional_embedding # [caption_len, model_dim]
      else:
        raise ValueError('not supported caption feature: {}'.format(self.config['caption_feature']))
      sequence = jnp.concatenate([projected_caption, sequence], axis = 0)
    mask = build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
    if caption_len > 0:
      mask = jnp.concatenate([data.embedding_mask, mask], axis = 0)
    mask = einshape('i->ji', mask, j = sequence.shape[0])
    mask = mask * basic_mask
    mask = einshape("ij->mij", mask, m = self.config['transformer']['n_heads']) # [n_heads, num * 2 * (cond_len + qoi_len), num * 2 * (cond_len + qoi_len)]
    sequence = self.transformer(sequence, mask = mask)
    sequence = self.post_projection(sequence) # [num * 2 * (cond_len + qoi_len), out_dim]
    return sequence
  
  def __call__(self, data): 
    '''
    this is for standard data shape, using prebuilt basic mask, index matrix, and out mask,
    caption included if config['caption_len'] > 0 (by default > 0)
    '''
    sequence = self.basic_forward(data, 'train', self.index_matrix_with_caption, self.basic_mask_with_caption,
                                  caption_len = self.config['caption_len'], shot_num_min = 0)
    sequence = sequence[self.out_mask_with_caption] # [num * qoi_len, out_dim]
    return sequence

  def forward_without_caption(self, data):
    '''
    this is for standard data shape, using prebuilt basic mask, index matrix, and out mask,
    caption not included
    '''
    sequence = self.basic_forward(data, 'train', self.index_matrix_without_caption, self.basic_mask_without_caption,
                                  caption_len = 0, shot_num_min = 1)
    sequence = sequence[self.out_mask_without_caption] # [num * qoi_len, out_dim]
    return sequence

  def predict(self, data, compact, caption_len):
    '''
    this is for flexible data shape, will build basic mask, index matrix, and out mask on the fly,
    used for prediction, i.e., only care about question
    '''
    shot_num_min = 0 # useless
    data_shape = tree.tree_map(lambda x: x.shape, data)
    basic_mask, index_matrix, out_mask = build_matrices_from_data_shape(data_shape, self.config, compact, 
                                                                        mode = 'test', caption_len = caption_len, shot_num_min = shot_num_min)
    sequence = self.basic_forward(data, 'test', index_matrix, basic_mask, caption_len = caption_len, shot_num_min = shot_num_min)
    sequence = sequence[-data.quest_qoi_mask.shape[-1]:,:] # [quest_qoi_len, out_dim]
    return sequence



def build_network_fn(data, key, config, return_model = False, compact = True, print_model = True):
  config = freeze(config)
  data = tree.tree_map(lambda x: x[0,0], data) # take off device and batch dimension
  data_shape = tree.tree_map(lambda x: x.shape, data)
  basic_mask_with_caption, index_matrix_with_caption, out_mask_with_caption = build_matrices_from_data_shape(data_shape, config, compact, 
                                                                                                mode = 'train', caption_len = config['caption_len'], shot_num_min = 0)
  basic_mask_without_caption, index_matrix_without_caption, out_mask_without_caption = build_matrices_from_data_shape(data_shape, config, compact, 
                                                                                           mode = 'train', caption_len = 0, shot_num_min = 1)

  model = IconGPTModel(config = config,
                       basic_mask_with_caption=basic_mask_with_caption, index_matrix_with_caption=index_matrix_with_caption, out_mask_with_caption=out_mask_with_caption,
                       basic_mask_without_caption=basic_mask_without_caption, index_matrix_without_caption=index_matrix_without_caption, out_mask_without_caption=out_mask_without_caption)
  
  subkey1, subkey2 = jax.random.split(key, 2)
  rngs = {'params': subkey1, 'dropout': subkey2}
  params = model.init(rngs, data)
  if print_model:
    print(utils.strip_ansi_codes(model.tabulate(rngs, data)))

  @jax.jit
  def forward_with_caption_fn(params, rng_key, data):
    return model.apply(params, data, rngs = {'dropout':rng_key})
  
  @jax.jit
  def forward_without_caption_fn(params, rng_key, data):
    return model.apply(params, data, rngs = {'dropout':rng_key}, method='forward_without_caption')
  
  @jax.jit
  def predict_with_caption_fn(params, rng_key, data):
    return model.apply(params, data, compact, config['caption_len'], rngs = {'dropout':rng_key}, method='predict') # [quest_qoi_len, out_dim]

  @jax.jit
  def predict_without_caption_fn(params, rng_key, data):
    return model.apply(params, data, compact, 0, rngs = {'dropout':rng_key}, method='predict') # [quest_qoi_len, out_dim]

  
  if return_model:
    return forward_with_caption_fn, forward_without_caption_fn, predict_with_caption_fn, predict_without_caption_fn, params, model
  else:
    return forward_with_caption_fn, forward_without_caption_fn, predict_with_caption_fn, predict_without_caption_fn, params




def test():
  from dataloader import DataProvider, print_eqn_caption
  from pprint import pprint
  import jax.tree_util as tree
  np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
  import haiku as hk
  rng = hk.PRNGSequence(42)

  run_config = utils.load_json('config_data/train_lm_input_id_config.json')
  run_config["demo_cond_len"] = 50
  run_config["demo_qoi_len"] = 60
  run_config["quest_cond_len"] = 70
  run_config["quest_qoi_len"] = 80
  data_provider = DataProvider(seed = 1, config = run_config,
                              file_names = '../icon/data/data0811/train*',
                              batch_size = 16, shuffle_buffer_size = 1000,)
  equation, caption, data, label =  data_provider.get_next_data()
  print_eqn_caption(equation, caption)
  print(tree.tree_map(lambda x: x.shape, data)) 

  model_config = utils.load_json('config_model/model_lm_input_id_config.json')
  model_config["demo_id_dim"] = 4
  forward_with_caption_fn, forward_without_caption_fn, predict_with_caption_fn, predict_without_caption_fn, params, model = \
    build_network_fn(data, next(rng), model_config, return_model = True, compact = True)
  
  fig = plot_model_consts(model.basic_mask_with_caption, model.index_matrix_with_caption, model.out_mask_with_caption)
  fig.savefig('mask_and_index_matrix_icon_gpt_forward_with_caption.pdf')

  fig = plot_model_consts(model.basic_mask_without_caption, model.index_matrix_without_caption, model.out_mask_without_caption)
  fig.savefig('mask_and_index_matrix_icon_gpt_forward_without_caption.pdf')

  basic_mask, index_matrix, out_mask = build_matrices_from_data_shape(data_shape = tree.tree_map(lambda x: x[0,0].shape, data),
                                                                      config = freeze(model_config), compact = True, 
                                                                      mode = 'test', caption_len = model_config['caption_len'], shot_num_min = 0)
  fig = plot_model_consts(basic_mask, index_matrix, out_mask)
  fig.savefig('mask_and_index_matrix_icon_gpt_predict_with_caption.pdf')

  basic_mask, index_matrix, out_mask = build_matrices_from_data_shape(data_shape = tree.tree_map(lambda x: x[0,0].shape, data),
                                                                      config = freeze(model_config), compact = True, 
                                                                      mode = 'test', caption_len = 0, shot_num_min = 1)
  fig = plot_model_consts(basic_mask, index_matrix, out_mask)
  fig.savefig('mask_and_index_matrix_icon_gpt_predict_without_caption.pdf')

  out = forward_with_caption_fn(params, next(rng), tree.tree_map(lambda x: x[0,0], data))
  print(out.shape)
  assert out.shape == (run_config['demo_num'] * run_config["demo_qoi_len"] + run_config["quest_qoi_len"], model_config['out_dim'])

  out = forward_without_caption_fn(params, next(rng), tree.tree_map(lambda x: x[0,0], data))
  print(out.shape)
  assert out.shape == ((run_config['demo_num'] - 1) * run_config["demo_qoi_len"] + run_config["quest_qoi_len"], model_config['out_dim'])


  quest_out = predict_with_caption_fn(params, next(rng), tree.tree_map(lambda x: x[0,0], data))
  print(quest_out.shape)
  assert quest_out.shape == (run_config["quest_qoi_len"], model_config['out_dim'])


  quest_out = predict_without_caption_fn(params, next(rng), tree.tree_map(lambda x: x[0,0], data))
  print(quest_out.shape)
  assert quest_out.shape == (run_config["quest_qoi_len"], model_config['out_dim'])


if __name__ == "__main__":
  test()