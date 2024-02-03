import numpy as np
import os
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from functools import partial
import tensorflow as tf
import numpy as np
import data_sequence
from pprint import pprint
import utils
import jax
import jax.tree_util as tree
import haiku as hk
from einshape import numpy_einshape as einshape
from collections import namedtuple
import dataloader_realtime as rt
from transformers import AutoTokenizer
from data_preparation.data_io import read_lists_from_file
import random
import importlib
from absl import flags


def print_eqn_caption(equation, caption, num = None, decode = False):
  if num is None:
    num = len(caption)
  else:
    num = min(num, len(caption))
  for inx in range(num):
    this_equation = equation[inx]
    this_caption = caption[inx]
    if decode:
      print('equation {}: {}'.format(inx, this_equation), flush=True)
      print('caption  {}: {}'.format(inx, this_caption), flush=True)
    else:
      print('equation {}: {}'.format(inx, repr(this_equation)), flush=True)
      print('caption  {}: {}'.format(inx, repr(this_caption)), flush=True)

Data = namedtuple('Data', ['input_id', 'embedding_raw', 'embedding_pool', 'embedding_mask',
                          'demo_cond_k', 'demo_cond_v', 'demo_cond_mask', 
                          'demo_qoi_k', 'demo_qoi_v', 'demo_qoi_mask',
                          'quest_cond_k', 'quest_cond_v', 'quest_cond_mask',
                          'quest_qoi_k', 'quest_qoi_mask',])

try:
  FLAGS = flags.FLAGS
  tf_rng = tf.random.Generator.from_seed(FLAGS.seed + 15)
  print("tf_rng from FLAGS, seed = {}".format(FLAGS.seed + 15), flush = True)
except:
  tf_rng = tf.random.Generator.from_seed(15)
  print("tf_rng from default, seed = {}".format(15), flush = True)

# define a parser function to parse the serialized example
def parse_function(example_proto, config):
    '''
    @return
      equation: string describing the equation
      caption: caption strings (n,)
      embedding_raw:  embedding of the caption strings, (n, len, embedding_dim)
      embedding_pool: pooled embedding of the caption strings, (n, embedding_dim)
      embedding_mask: mask of the caption strings, (n, len)
      cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
      cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
      qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
      qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
    '''
    feature_description = {
        'equation': tf.io.FixedLenFeature([], tf.string),
        'caption': tf.io.VarLenFeature(tf.string),
        'cond_k': tf.io.FixedLenFeature([], tf.string),
        'cond_v': tf.io.FixedLenFeature([], tf.string),
        'qoi_k': tf.io.FixedLenFeature([], tf.string),
        'qoi_v': tf.io.FixedLenFeature([], tf.string),
    }

    if "embedding" in config['load_list']:
      feature_description['embedding_raw'] = tf.io.FixedLenFeature([], tf.string)
      feature_description['embedding_pool'] = tf.io.FixedLenFeature([], tf.string)
      feature_description['embedding_mask'] = tf.io.FixedLenFeature([], tf.string)
    if "input_id" in config['load_list']:
      feature_description['input_id'] = tf.io.FixedLenFeature([], tf.string)
      feature_description['embedding_mask'] = tf.io.FixedLenFeature([], tf.string)

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    equation = parsed_example['equation']
    caption = tf.sparse.to_dense(parsed_example['caption'], default_value=b'') # tensor of shape (n, )
    cond_k = tf.io.parse_tensor(parsed_example['cond_k'], out_type=tf.float32)
    cond_v = tf.io.parse_tensor(parsed_example['cond_v'], out_type=tf.float32)
    qoi_k = tf.io.parse_tensor(parsed_example['qoi_k'], out_type=tf.float32)
    qoi_v = tf.io.parse_tensor(parsed_example['qoi_v'], out_type=tf.float32)

    caption_n = tf.shape(caption)[0]
    if "embedding" in config['load_list'] or "input_id" in config['load_list']:
      embedding_mask = tf.io.parse_tensor(parsed_example['embedding_mask'], out_type=tf.int64)
    else:
      embedding_mask = tf.zeros((caption_n, 0))
    
    if "embedding" in config['load_list']:
      embedding_raw = tf.io.parse_tensor(parsed_example['embedding_raw'], out_type=tf.float32)    
      embedding_pool = tf.io.parse_tensor(parsed_example['embedding_pool'], out_type=tf.float32)
    else:
      embedding_raw = tf.zeros((caption_n, 0, 0)) # dummy, shape (n, len, embedding_dim)
      embedding_pool = tf.zeros((caption_n, 0)) # dummy, shape (n, embedding_dim)

    if "input_id" in config['load_list']:
      input_id = tf.io.parse_tensor(parsed_example['input_id'], out_type=tf.int64)
    else:
      input_id = tf.zeros((caption_n, 0))

    return equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, cond_k, cond_v, qoi_k, qoi_v


def select_demo_quest(equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, cond_k, cond_v, qoi_k, qoi_v, config):
  demo_num = config['demo_num']

  if config['select_demo_quest'] == "random":
    num = tf.shape(cond_v)[0]
    demo_idx = tf_rng.uniform(shape = (demo_num,), minval = 0, maxval = num, dtype = tf.int32)
    demo_cond_k = tf.gather(cond_k, demo_idx)
    demo_cond_v = tf.gather(cond_v, demo_idx)
    demo_qoi_k = tf.gather(qoi_k, demo_idx)
    demo_qoi_v = tf.gather(qoi_v, demo_idx)

    quest_idx = tf_rng.uniform(shape = (1,), minval = 0, maxval = num, dtype = tf.int32)
    quest_cond_k = tf.gather(cond_k, quest_idx)
    quest_cond_v = tf.gather(cond_v, quest_idx)
    quest_qoi_k = tf.gather(qoi_k, quest_idx)
    quest_qoi_v = tf.gather(qoi_v, quest_idx)

  elif config['select_demo_quest'] == "ordered":
    demo_cond_k = cond_k[:demo_num,...]
    demo_cond_v = cond_v[:demo_num,...]
    demo_qoi_k = qoi_k[:demo_num,...]
    demo_qoi_v = qoi_v[:demo_num,...]

    quest_cond_k = cond_k[demo_num:demo_num+1,...]
    quest_cond_v = cond_v[demo_num:demo_num+1,...]
    quest_qoi_k = qoi_k[demo_num:demo_num+1,...]
    quest_qoi_v = qoi_v[demo_num:demo_num+1,...]

  else:
    raise ValueError(f"select must be random or ordered, but got {config['select_demo_quest']}")

  return equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, \
         demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, \
         quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v


def select_caption(equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, cond_k, cond_v, qoi_k, qoi_v, config):
  mode = config['select_caption']
  if mode == 'random':
    caption_idx = tf_rng.uniform(shape = (), minval = 0, maxval = tf.shape(caption)[0], dtype = tf.int32)
    caption = caption[caption_idx]
    input_id = input_id[caption_idx]
    embedding_raw = embedding_raw[caption_idx]
    embedding_pool = embedding_pool[caption_idx]
    embedding_mask = embedding_mask[caption_idx]
  elif mode == 'all':
    pass # do nothing, include all captions
  elif mode == 'random_dual':
    # randomly select one caption from first half, and one from second half
    caption_idx_1 = tf_rng.uniform(shape = (), minval = 0, maxval = tf.shape(caption)[0]//2, dtype = tf.int32)
    caption_idx_2 = tf_rng.uniform(shape = (), minval = tf.shape(caption)[0]//2, maxval = tf.shape(caption)[0], dtype = tf.int32)
    caption = tf.strings.join([caption[caption_idx_1], caption[caption_idx_2]], separator="||")
    input_id = tf.concat([input_id[caption_idx_1], input_id[caption_idx_2]], axis = -1)
    embedding_raw = tf.zeros((1,)) # dummy
    embedding_pool = tf.zeros((1,)) # dummy
    embedding_mask = tf.concat([embedding_mask[caption_idx_1], embedding_mask[caption_idx_2]], axis = -1)
  elif isinstance(mode, int):
    caption = caption[mode]
    input_id = input_id[mode]
    embedding_raw = embedding_raw[mode]
    embedding_pool = embedding_pool[mode]
    embedding_mask = embedding_mask[mode]
  else:
    raise ValueError('Invalid select_caption: {}'.format(mode))
  return equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, cond_k, cond_v, qoi_k, qoi_v


def build_feature(equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask,
                  demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  config):
  '''
  build the feature for each token
  pad the key and value to k_dim and v_dim
  @params
    return_raw: whether to return raw data
  @return
    raw: 0 if return_raw = False, otherwise, raw data
  '''
  raw = (demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v)

  if config['k_mode'] == 'naive':
    pass
  elif config['k_mode'] == 'itx':
    # index, time, space
    if tf.strings.regex_full_match(equation, "ode.*forward.*"):
      demo_cond_k = tf.stack([demo_cond_k[...,1],demo_cond_k[...,0]], axis = -1)  # index first, then time
      demo_qoi_k = tf.pad(demo_qoi_k, [[0,0],[0,0],[1,0]]) # add zero index
      quest_cond_k = tf.stack([quest_cond_k[...,1],quest_cond_k[...,0]], axis = -1) # index first, then time
      quest_qoi_k = tf.pad(quest_qoi_k, [[0,0],[0,0],[1,0]]) # add zero index
    elif tf.strings.regex_full_match(equation, "ode.*inverse.*") \
      or tf.strings.regex_full_match(equation, "series.*") \
      or tf.strings.regex_full_match(equation, "mfc_gparam.*"):
      # add zero index
      demo_cond_k = tf.pad(demo_cond_k, [[0,0],[0,0],[1,0]])
      demo_qoi_k = tf.pad(demo_qoi_k, [[0,0],[0,0],[1,0]])
      quest_cond_k = tf.pad(quest_cond_k, [[0,0],[0,0],[1,0]]) 
      quest_qoi_k = tf.pad(quest_qoi_k, [[0,0],[0,0],[1,0]])
    elif tf.strings.regex_full_match(equation, "pde.*") \
      or tf.strings.regex_full_match(equation, ".*weno.*"):
      # add zero index and time
      demo_cond_k = tf.pad(demo_cond_k, [[0,0],[0,0],[2,0]])
      demo_qoi_k = tf.pad(demo_qoi_k, [[0,0],[0,0],[2,0]])
      quest_cond_k = tf.pad(quest_cond_k, [[0,0],[0,0],[2,0]])
      quest_qoi_k = tf.pad(quest_qoi_k, [[0,0],[0,0],[2,0]])
    elif tf.strings.regex_full_match(equation, "mfc_rhoparam.*"):
      demo_cond_k = tf.pad(demo_cond_k, [[0,0],[0,0],[2,0]])
      demo_qoi_k = tf.pad(demo_qoi_k, [[0,0],[0,0],[1,0]])
      quest_cond_k = tf.pad(quest_cond_k, [[0,0],[0,0],[2,0]])
      quest_qoi_k = tf.pad(quest_qoi_k, [[0,0],[0,0],[1,0]])
    else:
      pass
  else:
    pass
  demo_cond_k = tf.pad(demo_cond_k, [[0,0],[0,0],[0, config['k_dim'] - tf.shape(demo_cond_k)[-1]]])
  demo_cond_v = tf.pad(demo_cond_v, [[0,0],[0,0],[0, config['v_dim'] - tf.shape(demo_cond_v)[-1]]])
  demo_qoi_k = tf.pad(demo_qoi_k, [[0,0],[0,0],[0, config['k_dim'] - tf.shape(demo_qoi_k)[-1]]])
  demo_qoi_v = tf.pad(demo_qoi_v, [[0,0],[0,0],[0, config['v_dim'] - tf.shape(demo_qoi_v)[-1]]])
  quest_cond_k = tf.pad(quest_cond_k, [[0,0],[0,0],[0, config['k_dim'] - tf.shape(quest_cond_k)[-1]]])
  quest_cond_v = tf.pad(quest_cond_v, [[0,0],[0,0],[0, config['v_dim'] - tf.shape(quest_cond_v)[-1]]])
  quest_qoi_k = tf.pad(quest_qoi_k, [[0,0],[0,0],[0, config['k_dim'] - tf.shape(quest_qoi_k)[-1]]])
  quest_qoi_v = tf.pad(quest_qoi_v, [[0,0],[0,0],[0, config['v_dim'] - tf.shape(quest_qoi_v)[-1]]])
  if config['return_raw']:
    return raw, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v
  else:
    return 0, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v

def build_sequence(raw, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask,
                  demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v,
                  config):
  '''
  build the sequence,
  @param:
    raw: raw data, see build_feature
    equation: string describing the equation
    other parameters: after building features, but raw in sequence dimension
    config: config dict
  @return:
    see build_prompt_and_mask input
    quest_qoi_k: list of [qoi_len, k_dim]
    quest_qoi_v: list of [qoi_len, v_dim]
    quest_qoi_mask: list of [qoi_len]
  '''
  if tf.strings.regex_full_match(equation, "pde.*spatial_forward.*"): 
    build_fn = data_sequence.build_pde_spatial_forward
    this_config = config['pde_spatial_forward']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  config, this_config)
  elif tf.strings.regex_full_match(equation, "pde.*spatial_inverse.*"): 
    build_fn = data_sequence.build_pde_spatial_inverse
    this_config = config['pde_spatial_inverse']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  config, this_config)
  elif tf.strings.regex_full_match(equation, "ode.*forward.*"):
    build_fn = data_sequence.build_ode_forward
    this_config = config['ode_forward']   
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  config, this_config)
  elif tf.strings.regex_full_match(equation, "ode.*inverse.*"): 
    # ode inverse, drop the last qoi (control)
    build_fn = data_sequence.build_ode_inverse
    this_config = config['ode_inverse']  
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  config, this_config)
  elif tf.strings.regex_full_match(equation, "series.*"):
    build_fn = data_sequence.build_time_series
    this_config = config['time_series'] 
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  config, this_config)
  elif tf.strings.regex_full_match(equation, "mfc_gparam.*forward.*"):
    build_fn = data_sequence.build_mfc_gparam_forward
    this_config = config['mfc_gparam_forward']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v,
                  config, this_config)
  elif tf.strings.regex_full_match(equation, "mfc_rhoparam.*forward.*"):
    build_fn = data_sequence.build_mfc_rhoparam_forward
    this_config = config['mfc_rhoparam_forward']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v,
                  config, this_config)
  else: # other problems
    if not tf.strings.regex_full_match(equation, ".*weno.*"):
      tf.print("WARNING: see other problems!")
    build_fn = data_sequence.build_others
    this_config = config['others']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                      quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                      config, this_config)

  equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, \
        quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
        demo_cond_mask, demo_qoi_mask, quest_cond_mask, quest_qoi_mask = out
  
  return raw, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, \
        demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, \
        quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
        demo_cond_mask, demo_qoi_mask, quest_cond_mask, quest_qoi_mask

def build_model_input(raw, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask,
                      demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                      quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v,
                      demo_cond_mask, demo_qoi_mask, quest_cond_mask, quest_qoi_mask, 
                      config):
  '''
  concatenate the inputs (demos and condition), add index, and apply masks, to build prompt and prompt_mask
  @params:
    raw: 0 if return_raw = False, otherwise, raw data
    equation: string describing the equation
    demo_cond_k: list of array [cond_len, k_dim], len = demo_num
    demo_cond_v: list of array [cond_len, v_dim], len = demo_num
    demo_qoi_k: list of array [qoi_len, k_dim], len = demo_num
    demo_qoi_v: list of array [qoi_len, v_dim], len = demo_num
    quest_cond_k: list of array [cond_len, k_dim], len = quest_num
    quest_cond_v: list of array [cond_len, v_dim], len = quest_num
    quest_qoi_k: list of array [qoi_len, k_dim], len = quest_num
    quest_qoi_v: list of array [qoi_len, v_dim], len = quest_num
    demo_cond_mask: list of [cond_len], len = demo_num
    demo_qoi_mask: list of [qoi_len], len = demo_num
    quest_cond_mask: list of [cond_len], len = quest_num
    quest_qoi_mask: list of [qoi_len], len = quest_num
  @return:
    raw: 0 if return_raw = False, otherwise, raw data
    equation: string describing the equation
    demo_cond_k: array [demo_num, cond_len, k_dim]
    demo_cond_v: array [demo_num, cond_len, v_dim]
    demo_cond_mask: array [demo_num, cond_len]
    demo_qoi_k: array [demo_num, qoi_len, k_dim]
    demo_qoi_v: array [demo_num, qoi_len, v_dim]
    demo_qoi_mask: array [demo_num, qoi_len]
    quest_cond_k: array [quest_num, cond_len, k_dim]
    quest_cond_v: array [quest_num, cond_len, v_dim]
    quest_cond_mask: array [quest_num, cond_len]
    quest_qoi_k: array [quest_num, qoi_len, k_dim]
    quest_qoi_v: array [quest_num, qoi_len, v_dim]
    quest_qoi_mask: array [quest_num, qoi_len]
  @return:
  '''
  demo_cond_mask = tf.convert_to_tensor(demo_cond_mask) # [demo_num, demo_cond_len]
  demo_cond_k = tf.convert_to_tensor(demo_cond_k) * tf.cast(demo_cond_mask, tf.float32)[...,None] # [demo_num, demo_cond_len, k_dim]
  demo_cond_v = tf.convert_to_tensor(demo_cond_v) * tf.cast(demo_cond_mask, tf.float32)[...,None] # [demo_num, demo_cond_len, v_dim]

  demo_qoi_mask = tf.convert_to_tensor(demo_qoi_mask) # [demo_num, demo_qoi_len]
  demo_qoi_k = tf.convert_to_tensor(demo_qoi_k) * tf.cast(demo_qoi_mask, tf.float32)[...,None] # [demo_num, demo_qoi_len, k_dim]
  demo_qoi_v = tf.convert_to_tensor(demo_qoi_v) * tf.cast(demo_qoi_mask, tf.float32)[...,None] # [demo_num, demo_qoi_len, v_dim]

  quest_cond_mask = tf.convert_to_tensor(quest_cond_mask) # [quest_num, quest_cond_len]
  quest_cond_k = tf.convert_to_tensor(quest_cond_k) * tf.cast(quest_cond_mask, tf.float32)[...,None] # [quest_num, quest_cond_len, k_dim]
  quest_cond_v = tf.convert_to_tensor(quest_cond_v) * tf.cast(quest_cond_mask, tf.float32)[...,None] # [quest_num, quest_cond_len, v_dim]

  quest_qoi_mask = tf.convert_to_tensor(quest_qoi_mask) # [quest_num, quest_qoi_len]
  quest_qoi_k = tf.convert_to_tensor(quest_qoi_k) * tf.cast(quest_qoi_mask, tf.float32)[...,None] # [quest_num, quest_qoi_len, k_dim]
  quest_qoi_v = tf.convert_to_tensor(quest_qoi_v) * tf.cast(quest_qoi_mask, tf.float32)[...,None] # [quest_num, quest_qoi_len, v_dim]
  
  return raw, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, \
        demo_cond_k, demo_cond_v, demo_cond_mask, \
        demo_qoi_k, demo_qoi_v, demo_qoi_mask, \
        quest_cond_k, quest_cond_v, quest_cond_mask, \
        quest_qoi_k, quest_qoi_v, quest_qoi_mask

def get_tf_dataset(seed, config, file_names, 
                    batch_size, shuffle_buffer_size, deterministic, 
                    drop_remainder, shuffle_dataset,
                    num_epochs, num_parallel_calls, real_time):
    '''
    @params:
      seed: int
      config: dict, config for different problems
      file_names: string or list of strings
      batch_size: int
      shuffle_buffer_size: int
      deterministic: bool, whether to use deterministic dataset
      return_raw: bool, whether to return raw data. if false, use 0 as a placeholder for raw data
                  note that different dataset have different raw data size,
                  therefore, it's better to set batch_size = 1 when return_raw, espetially for multiple datasets
      drop_remainer: bool, if drop the remainer of dataset in the end of each epoch
      shuffle_dataset: bool, whether to shuffle the dataset
      num_epochs: int or None (no repeat)
      num_parallel_calls: for tf.data.Dataset
    @return:
      tf_dataset: tf.data.Dataset
    '''
    filenames_dataset = tf.data.Dataset.list_files(file_names, seed = seed + 1, shuffle = shuffle_dataset)
    dataset = filenames_dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls = num_parallel_calls)
    
    if real_time:
      dataset = dataset.map(partial(rt.parse_function, config = config), num_parallel_calls = num_parallel_calls)
      dataset = dataset.map(partial(rt.select_caption, config = config), num_parallel_calls = num_parallel_calls)
    else:
      dataset = dataset.map(partial(parse_function, config = config), num_parallel_calls = num_parallel_calls)
      dataset = dataset.map(partial(select_caption, config = config), num_parallel_calls = num_parallel_calls)
    
    dataset = dataset.map(partial(select_demo_quest, config = config), num_parallel_calls = num_parallel_calls)
    dataset = dataset.map(partial(build_feature, config = config), num_parallel_calls = num_parallel_calls)
    dataset = dataset.map(partial(build_sequence, config = config), num_parallel_calls = num_parallel_calls)
    dataset = dataset.map(partial(build_model_input, config = config), num_parallel_calls = num_parallel_calls)
    
    if shuffle_dataset:
      dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed = seed + 2)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder, num_parallel_calls = num_parallel_calls)
    dataset = dataset.prefetch(buffer_size = 1)
    dataset = dataset.repeat(num_epochs)

    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    dataset = dataset.with_options(options)

    return dataset

class DataProvider():
  def __init__(self, seed, config, file_names,
               batch_size, shuffle_buffer_size, 
               deterministic = True,
               drop_remainder = True, shuffle_dataset = True,
               num_epochs = None,
               num_devices = len(jax.devices()), 
               caption_home_dir = "data_preparation", # in case run from other directory
               name = 'DataProvider',
               card = 'gpt2',
               real_time = False):
    '''
    data provider
    '''
    self.seed = seed
    self.num_devices = num_devices
    self.name = name
    self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    self.config = config
    self.real_time = real_time

    if deterministic:
      num_parallel_calls = None
    else:
      num_parallel_calls = tf.data.AUTOTUNE
    self.dataset = get_tf_dataset(
                             seed = seed, config = config, file_names = file_names, 
                             batch_size = batch_size, 
                             shuffle_buffer_size = shuffle_buffer_size, 
                             deterministic = deterministic, 
                             drop_remainder = drop_remainder,
                             shuffle_dataset = shuffle_dataset,
                             num_epochs = num_epochs, 
                             num_parallel_calls = num_parallel_calls,
                             real_time = real_time)
    
    self.dataset = iter(self.dataset)

    # take care of caption
    if self.real_time and ('input_id' in self.config['load_list']):
      self.caption_home_dir = caption_home_dir
      # handle tokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(card)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      # handle caption dict
      self.caption_dir = config['caption_dir']
      self.caption_dict = self.load_caption(config['load_caption'])
      # handle caption suffix
      self.caption_suffix = utils.load_json('{}/{}/suffix.json'.format(caption_home_dir,self.caption_dir))
      # handle fill_caption_number
      module_name = f"data_preparation.{self.caption_dir}.resolve"
      module = importlib.import_module(module_name)
      self.fill_caption_number = getattr(module, 'fill_caption_number')

  def tokenize(self, text_list, max_length):
    encoded_inputs = self.tokenizer(text_list, return_tensors='np', padding='max_length', max_length=max_length, truncation=True)
    if encoded_inputs['attention_mask'][:,-1].max() == 1:
      encoded_inputs_untruncated = self.tokenizer(text_list, return_tensors='np', padding='max_length')
      needed_length = np.sum(encoded_inputs_untruncated['attention_mask'], axis = -1).max()
      print('WARNING: max_length = {} may not be enough, need at least {}'.format(max_length, needed_length))
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask']

  def load_caption(self, mode):
    file_dict = {
      "ode_auto_const_forward": "ode1",
      "ode_auto_const_inverse": "ode1",
      "ode_auto_linear1_forward": "ode2",
      "ode_auto_linear1_inverse": "ode2",
      "ode_auto_linear2_forward": "ode3",
      "ode_auto_linear2_inverse": "ode3",
      "series_damped_oscillator_forward": "series",
      "series_damped_oscillator_inverse": "series",
      "pde_poisson_spatial_forward": "pde1",
      "pde_poisson_spatial_inverse": "pde1",
      "pde_porous_spatial_forward": "pde2",
      "pde_porous_spatial_inverse": "pde2",
      "pde_cubic_spatial_forward": "pde3",
      "pde_cubic_spatial_inverse": "pde3",
      "mfc_gparam_hj_forward11": "mfc_gparam",
      "mfc_gparam_hj_forward12": "mfc_gparam",
      "mfc_gparam_hj_forward22": "mfc_gparam",
      "mfc_rhoparam_hj_forward11": "mfc_rhoparam",
      "mfc_rhoparam_hj_forward12": "mfc_rhoparam",
    }

    caption_dict = {}
    print("loading caption for dataset {} from {}".format(self.name, self.caption_dir))
    for key, value in file_dict.items():
      try:
        caption_dict[key] = read_lists_from_file('{}/{}/{}.md'.format(self.caption_home_dir, self.caption_dir, value), mode)
        print('INFO: {} : {} found'.format(key, '{}/{}/{}.md'.format(self.caption_home_dir, self.caption_dir, value)))
      except FileNotFoundError:
        print('WARNING: {} : {} not found'.format(key, '{}/{}/{}.md'.format(self.caption_home_dir, self.caption_dir, value)))
        caption_dict[key] = []
    print("caption loaded", flush=True)
    return caption_dict
  

  def add_caption_suffix(self, caption, identifier):
    caption = caption + self.caption_suffix[identifier]
    return caption
  
          
  def get_caption(self, equation, mode):
    equation = equation.numpy().decode('utf-8')
    identifier, params = rt.split_equation(equation)
    if mode == 'random':
      caption_list = self.caption_dict[identifier] # should be a list of captions
      caption_template = random.choice(caption_list)
      caption = self.fill_caption_number(caption_template, identifier, params)
      caption = self.add_caption_suffix(caption, identifier)
    else:
      raise ValueError(f'caption mode = {mode} not supported')
    
    return caption

  def get_next_data(self, return_raw = False, caption_max_len = 300):
    raw, equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, \
    demo_cond_k, demo_cond_v, demo_cond_mask, \
    demo_qoi_k, demo_qoi_v, demo_qoi_mask, \
    quest_cond_k, quest_cond_v, quest_cond_mask, \
    quest_qoi_k, quest_qoi_v, quest_qoi_mask = next(self.dataset)
    
    input_id = input_id.numpy()
    embedding_raw = embedding_raw.numpy()
    embedding_pool = embedding_pool.numpy()
    embedding_mask = embedding_mask.numpy().astype(bool)
    demo_cond_k = demo_cond_k.numpy()
    demo_cond_v = demo_cond_v.numpy()
    demo_cond_mask = demo_cond_mask.numpy().astype(bool)
    demo_qoi_k = demo_qoi_k.numpy()
    demo_qoi_v = demo_qoi_v.numpy()
    demo_qoi_mask = demo_qoi_mask.numpy().astype(bool)
    quest_cond_k = quest_cond_k.numpy()
    quest_cond_v = quest_cond_v.numpy()
    quest_cond_mask = quest_cond_mask.numpy().astype(bool)
    quest_qoi_k = quest_qoi_k.numpy()
    quest_qoi_v = quest_qoi_v.numpy()
    quest_qoi_mask = quest_qoi_mask.numpy().astype(bool)

    if self.real_time and ('input_id' in self.config['load_list']):
      caption = [self.get_caption(eqn, self.config['select_caption']) for eqn in equation] # list of strings
      input_id, embedding_mask = self.tokenize(caption, caption_max_len) # both [batch_size, caption_max_len]

    data = Data(input_id = input_id, 
                embedding_raw = embedding_raw, 
                embedding_pool = embedding_pool, 
                embedding_mask = embedding_mask, 
                demo_cond_k = demo_cond_k, 
                demo_cond_v = demo_cond_v, 
                demo_cond_mask = demo_cond_mask, 
                demo_qoi_k = demo_qoi_k,
                demo_qoi_v = demo_qoi_v,
                demo_qoi_mask = demo_qoi_mask,
                quest_cond_k = quest_cond_k,
                quest_cond_v = quest_cond_v,
                quest_cond_mask = quest_cond_mask,
                quest_qoi_k = quest_qoi_k,
                quest_qoi_mask = quest_qoi_mask,
                )
    label = quest_qoi_v
    
    if self.num_devices > 0: # split data into multiple devices
      data = tree.tree_map(lambda x: einshape('(ij)...->ij...', x, i = self.num_devices), data)
      label = einshape('(ij)...->ij...', label, i = self.num_devices)
  
    if return_raw:
      return raw, equation, caption, data, label
    else:
      return equation, caption, data, label

def split_data(caption, data, demo_num_list, caption_id_list = (-1,)):
  # This function mainly splits examples
  # caption_id_list is just a counter, only its length matters
  for demo_num in demo_num_list:
    for ci in caption_id_list:
      new_data = data._replace(demo_cond_k = data.demo_cond_k[..., :demo_num, :, :],
                              demo_cond_v = data.demo_cond_v[..., :demo_num, :, :],
                              demo_cond_mask = data.demo_cond_mask[..., :demo_num, :], 
                              demo_qoi_k = data.demo_qoi_k[..., :demo_num, :, :],
                              demo_qoi_v = data.demo_qoi_v[..., :demo_num, :, :],
                              demo_qoi_mask = data.demo_qoi_mask[..., :demo_num, :])
      new_caption = caption
      yield demo_num, ci, new_caption, new_data


def test(file_names, config):

  print('==================train data====================')
  data_provider = DataProvider(seed = 1, config = config, 
                              file_names = file_names, num_devices = 0,
                              batch_size = 16, shuffle_buffer_size = 1000,
                              real_time = True)
  equation, caption, data, label =  data_provider.get_next_data()
  print_eqn_caption(equation, caption)
  pprint(tree.tree_map(lambda x: (type(x), x.shape), data)._asdict()) 

  print('==================split data====================')
  for demo_num, ci, new_caption, new_data in split_data(caption, data, demo_num_list = [0]):
    print(demo_num, ci)
    print_eqn_caption(equation, new_caption, num = 1)
    pprint(tree.tree_map(lambda x: x.shape, new_data)._asdict())
    try:
      print(new_data.input_id[0,:])
      print(new_data.embedding_mask[0,:])
    except:
      pass 
    print('======================================')

  print(f'==================test {file_names} passed ====================')
  for i in range(10):
    utils.timer.tic("select_caption {}".format(config['select_caption']))
    data_provider.get_next_data()
    utils.timer.toc("select_caption {}".format(config['select_caption']))


if __name__ == "__main__":
    

    np.set_printoptions(threshold=np.inf, precision=3, suppress=True)

    config = utils.load_json('config_data/train_lm_config.json')
    file_names = '/home/shared/icon/data/data0910c/train*'
    config['load_list'] = []
    test(file_names, config)

    config['load_list'] = ['input_id']
    config['select_caption'] = 'random'
    test(file_names, config)

    # config['load_list'] = ['input_id']
    # config['select_caption'] = 'random_dual'
    # test(file_names, config)