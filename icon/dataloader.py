import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
from functools import partial
import tensorflow as tf
import numpy as np
import data_sequence
from pprint import pprint

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import haiku as hk
from einshape import jax_einshape as einshape

tf_rng = tf.random.Generator.from_seed(15)

# define a parser function to parse the serialized example
def parse_function(example_proto):
    '''
    @return
      equation: string describing the equation
      cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
      cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
      qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
      qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
    '''
    feature_description = {
        'equation': tf.io.FixedLenFeature([], tf.string),
        'cond_k': tf.io.FixedLenFeature([], tf.string),
        'cond_v': tf.io.FixedLenFeature([], tf.string),
        'qoi_k': tf.io.FixedLenFeature([], tf.string),
        'qoi_v': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    equation = parsed_example['equation']
    cond_k = tf.io.parse_tensor(parsed_example['cond_k'], out_type=tf.float32)
    cond_v = tf.io.parse_tensor(parsed_example['cond_v'], out_type=tf.float32)
    qoi_k = tf.io.parse_tensor(parsed_example['qoi_k'], out_type=tf.float32)
    qoi_v = tf.io.parse_tensor(parsed_example['qoi_v'], out_type=tf.float32)
    return equation, cond_k, cond_v, qoi_k, qoi_v


# random select demos and the question
def random_select(equation, cond_k, cond_v, qoi_k, qoi_v, demo_num):
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
  return equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v

# ordered select demos and the question
def ordered_select(equation, cond_k, cond_v, qoi_k, qoi_v, demo_num):
  demo_cond_k = cond_k[:demo_num,...]
  demo_cond_v = cond_v[:demo_num,...]
  demo_qoi_k = qoi_k[:demo_num,...]
  demo_qoi_v = qoi_v[:demo_num,...]

  quest_cond_k = cond_k[demo_num:demo_num+1,...]
  quest_cond_v = cond_v[demo_num:demo_num+1,...]
  quest_qoi_k = qoi_k[demo_num:demo_num+1,...]
  quest_qoi_v = qoi_v[demo_num:demo_num+1,...]
  return equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v


def build_feature(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, k_dim, v_dim, k_mode, return_raw):
  '''
  build the feature for each token
  pad the key and value to k_dim and v_dim
  '''
  raw = (demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v)

  if k_mode == 'naive':
    pass
  elif k_mode == 'itx':
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
    elif tf.strings.regex_full_match(equation, "pde.*"):
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
  demo_cond_k = tf.pad(demo_cond_k, [[0,0],[0,0],[0, k_dim - tf.shape(demo_cond_k)[-1]]])
  demo_cond_v = tf.pad(demo_cond_v, [[0,0],[0,0],[0, v_dim - tf.shape(demo_cond_v)[-1]]])
  demo_qoi_k = tf.pad(demo_qoi_k, [[0,0],[0,0],[0, k_dim - tf.shape(demo_qoi_k)[-1]]])
  demo_qoi_v = tf.pad(demo_qoi_v, [[0,0],[0,0],[0, v_dim - tf.shape(demo_qoi_v)[-1]]])
  quest_cond_k = tf.pad(quest_cond_k, [[0,0],[0,0],[0, k_dim - tf.shape(quest_cond_k)[-1]]])
  quest_cond_v = tf.pad(quest_cond_v, [[0,0],[0,0],[0, v_dim - tf.shape(quest_cond_v)[-1]]])
  quest_qoi_k = tf.pad(quest_qoi_k, [[0,0],[0,0],[0, k_dim - tf.shape(quest_qoi_k)[-1]]])
  quest_qoi_v = tf.pad(quest_qoi_v, [[0,0],[0,0],[0, v_dim - tf.shape(quest_qoi_v)[-1]]])
  if return_raw:
    return raw, equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v
  else:
    return 0, equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v

def build_sequence(raw, equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, demo_num, cond_len, qoi_len, config):
  '''
  build the sequence,
  @param:
    equation: string describing the equation
    other parameters: after building features, but raw in sequence dimension
    seed: int random seed, reserve 100 for each problem
    demo_num: number of demos
    cond_len: max condition length
    qoi_len: max qoi length
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
                  demo_num, cond_len, qoi_len, this_config)
  elif tf.strings.regex_full_match(equation, "pde.*spatial_inverse.*"): 
    build_fn = data_sequence.build_pde_spatial_inverse
    this_config = config['pde_spatial_inverse']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  demo_num, cond_len, qoi_len, this_config)
  elif tf.strings.regex_full_match(equation, "ode.*forward.*"):
    build_fn = data_sequence.build_ode_forward
    this_config = config['ode_forward']   
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  demo_num, cond_len, qoi_len, this_config)
  elif tf.strings.regex_full_match(equation, "ode.*inverse.*"): 
    # ode inverse, drop the last qoi (control)
    build_fn = data_sequence.build_ode_inverse
    this_config = config['ode_inverse']  
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  demo_num, cond_len, qoi_len, this_config)
  elif tf.strings.regex_full_match(equation, "series.*"):
    build_fn = data_sequence.build_time_series
    this_config = config['time_series'] 
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                  demo_num, cond_len, qoi_len, this_config)
  elif tf.strings.regex_full_match(equation, "mfc_gparam.*forward.*"):
    build_fn = data_sequence.build_mfc_gparam_forward
    this_config = config['mfc_gparam_forward']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v,
                  demo_num, cond_len, qoi_len, this_config)
  elif tf.strings.regex_full_match(equation, "mfc_rhoparam.*forward.*"):
    build_fn = data_sequence.build_mfc_rhoparam_forward
    this_config = config['mfc_rhoparam_forward']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v,
                  demo_num, cond_len, qoi_len, this_config)
  else: # other problems
    tf.print("WARNING: see other problems!")
    build_fn = data_sequence.build_others
    this_config = config['others']
    out = build_fn(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                      quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                      demo_num, cond_len, qoi_len, this_config)
  return raw, *out

def build_prompt_and_mask(demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v,
                          demo_cond_mask, demo_qoi_mask, quest_cond_mask, demo_num, cond_len, qoi_len):
  '''
  concatenate the inputs (demos and condition), add index, and apply masks, to build prompt and prompt_mask
  @params:
    demo_cond_k: post-processed list of array [cond_len, k_dim]
    demo_cond_v: post-processed list of array [cond_len, v_dim]
    demo_qoi_k: post-processed list of array [qoi_len, k_dim]
    demo_qoi_v: post-processed list of array [qoi_len, v_dim]
    quest_cond_k: post-processed list of array [cond_len, k_dim]
    quest_cond_v: post-processed list of array [cond_len, v_dim]
    demo_cond_mask: list of [cond_len]
    demo_qoi_mask: list of [qoi_len]
    quest_cond_mask: list of [cond_len]
  @return:
    prompt: 2D array
    prompt_mask: 1D array
  '''

  prompt_list = []

  for i in range(demo_num):
    demo_cond_i_index = tf.one_hot(i*tf.ones((cond_len,), dtype=tf.int32), demo_num + 1) # [cond_len, demo_num + 1]
    demo_cond_i = tf.concat([demo_cond_k[i], demo_cond_v[i], demo_cond_i_index], axis = -1) # [cond_len, k_dim + v_dim + demo_num + 1]
    demo_qoi_i_index = - tf.one_hot(i*tf.ones((qoi_len,), dtype=tf.int32), demo_num + 1) # [qoi_len, demo_num + 1]
    demo_qoi_i = tf.concat([demo_qoi_k[i], demo_qoi_v[i], demo_qoi_i_index], axis = -1) # [qoi_len, k_dim + v_dim + demo_num + 1]
    prompt_list.append(demo_cond_i)
    prompt_list.append(demo_qoi_i)

  quest_cond_index = tf.one_hot(demo_num*tf.ones((cond_len,), dtype=tf.int32), demo_num + 1) # [cond_len, demo_num + 1]
  quest_cond = tf.concat([quest_cond_k[0], quest_cond_v[0], quest_cond_index], axis = -1) # [cond_len, k_dim + v_dim + demo_num + 1]
  prompt_list.append(quest_cond)
  prompt = tf.concat(prompt_list, axis = 0) # [demo_num * (cond_len + qoi_len) + cond_len, k_dim + v_dim + demo_num + 1]
  
  prompt_mask = []
  for i in range(demo_num):
    prompt_mask.append(demo_cond_mask[i])
    prompt_mask.append(demo_qoi_mask[i])
  prompt_mask.append(quest_cond_mask[0])
  prompt_mask = tf.concat(prompt_mask, axis = 0) # [demo_num * (cond_len + qoi_len) + cond_len]

  # should be optional, since we will apply mask in the model, but I want to make it clear in prompt
  prompt = prompt * tf.cast(prompt_mask, tf.float32)[:,None] # [demo_num * (cond_len + qoi_len) + cond_len, k_dim + v_dim + demo_num + 1]
  return prompt, prompt_mask
    
def build_model_input(raw, equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, \
                      quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
                      demo_cond_mask, demo_qoi_mask, quest_cond_mask, quest_qoi_mask, 
                      demo_num, cond_len, qoi_len):
    prompt, prompt_mask = build_prompt_and_mask(demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v,
                          demo_cond_mask, demo_qoi_mask, quest_cond_mask, demo_num, cond_len, qoi_len)
    query = quest_qoi_k[0]  # [qoi_len, k_dim]
    ground_truth = quest_qoi_v[0]  # [qoi_len, v_dim]
    query_mask = quest_qoi_mask[0]  # [qoi_len]
    return raw, equation, prompt, prompt_mask, query, query_mask, ground_truth

def get_tf_dataset(seed, file_names, batch_size, num_epochs, shuffle_buffer_size, drop_remainder,
                   demo_num, cond_len, qoi_len, k_dim, v_dim, config, select, k_mode, return_raw,
                   num_parallel_calls = tf.data.AUTOTUNE, deterministic = False, shuffle_dataset = True):
    '''
    @params:
      seed: int
      file_names: string or list of strings
      batch_size: int
      num_epochs: int or None (no repeat)
      shuffle_buffer_size: int
      demo_num: int, total number of demos
      cond_len: int, total length of condition
      qoi_len: int, total length of qoi
      k_dim: int, dimension of key
      v_dim: int, dimension of value
      config: dict, config for different problems
      select: string, "random" or "ordered", indicating the rule for demo/quest selection
      k_mode: string, the mode for key
      return_raw: bool, whether to return raw data. if false, use 0 as a placeholder for raw data
                  note that different dataset have different raw data size,
                  therefore, it's better to set batch_size = 1 when return_raw, espetially for multiple datasets
      shuffle_dataset: bool, whether to shuffle the dataset
    @return:
      tf_dataset: tf.data.Dataset
    '''
    filenames_dataset = tf.data.Dataset.list_files(file_names, seed = seed + 1, shuffle = shuffle_dataset)
    dataset = filenames_dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls = num_parallel_calls)
    dataset = dataset.map(parse_function)
    # dataset = dataset.cache() # should be placed before adding randomness

    # equation, cond_k, cond_v, qoi_k, qoi_v
    if select == "random":
      dataset = dataset.map(partial(random_select, demo_num = demo_num), num_parallel_calls = num_parallel_calls)
    elif select == "ordered":
      dataset = dataset.map(partial(ordered_select, demo_num = demo_num), num_parallel_calls = num_parallel_calls)
    else:
      raise ValueError("select must be random or ordered")
    # equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
    #           quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v
    dataset = dataset.map(partial(build_feature, k_dim = k_dim, v_dim = v_dim, k_mode = k_mode, return_raw = return_raw), num_parallel_calls = num_parallel_calls)
    # equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, 
    #           quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v
    dataset = dataset.map(partial(build_sequence, demo_num = demo_num, cond_len = cond_len, qoi_len = qoi_len, config = config), 
                          num_parallel_calls = num_parallel_calls)
    # equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v, \
    #           quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
    #           demo_cond_mask, demo_qoi_mask, quest_cond_mask, quest_qoi_mask
    dataset = dataset.map(partial(build_model_input, demo_num = demo_num, cond_len = cond_len, qoi_len = qoi_len), 
                          num_parallel_calls = num_parallel_calls)
    # equation, prompt, prompt_mask, query, query_mask, ground_truth
    
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
  def __init__(self, seed, demo_num, cond_len, qoi_len,
               batch_size, shuffle_buffer_size, file_names,
               k_dim, v_dim, config, select, k_mode,
               deterministic, return_raw = False,
               drop_remainder = True, shuffle_dataset = True,
               num_epochs = None,
               num_devices = len(jax.devices()), 
               name = 'DataProvider'):
    '''
    data provider
    '''
    self.seed = seed
    self.num_devices = num_devices
    self.name = name
    self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    if deterministic:
      num_parallel_calls = None
    else:
      num_parallel_calls = tf.data.AUTOTUNE
    self.dataset = get_tf_dataset(
                             seed = seed, file_names = file_names, 
                             batch_size = batch_size, num_epochs = num_epochs, 
                             shuffle_buffer_size = shuffle_buffer_size, 
                             drop_remainder = drop_remainder,
                             demo_num = demo_num, cond_len = cond_len, qoi_len = qoi_len,
                             k_dim = k_dim, v_dim = v_dim,
                             config = config, select = select, k_mode = k_mode,
                             return_raw = return_raw,
                             num_parallel_calls = num_parallel_calls,
                             deterministic = deterministic, shuffle_dataset = shuffle_dataset)
    
    self.dataset = iter(self.dataset)

  def get_next_data(self, decode_equation = False, list_size = 0, return_raw = False):
    raw, equation, prompt, mask, query, query_mask, ground_truth = next(self.dataset)
    if decode_equation:
      equation = [s.decode('utf-8') for s in equation.numpy()]
    
    if list_size > 0: # actually never used...
      prompt = jnp.split(einshape('(ij)...->ij...', prompt.numpy(), i = self.num_devices * list_size), list_size, axis = 0)
      mask = jnp.split(einshape('(ij)...->ij...', mask.numpy(), i = self.num_devices * list_size), list_size, axis = 0)
      query = jnp.split(einshape('(ij)...->ij...', query.numpy(), i = self.num_devices * list_size), list_size, axis = 0)
      query_mask = jnp.split(einshape('(ij)...->ij...', query_mask.numpy(), i = self.num_devices * list_size), list_size, axis = 0)
      ground_truth = jnp.split(einshape('(ij)...->ij...', ground_truth.numpy(), i = self.num_devices * list_size), list_size, axis = 0)
    else: # this is the default case
      prompt = einshape('(ij)...->ij...', prompt.numpy(), i = self.num_devices) # (num_devices, batch_size, prompt_length, prompt_dim)
      mask = einshape('(ij)...->ij...', mask.numpy(), i = self.num_devices) # (num_devices, batch_size, prompt_length)
      query = einshape('(ij)...->ij...', query.numpy(), i = self.num_devices) # (num_devices, batch_size, query_length, key_dim)
      query_mask = einshape('(ij)...->ij...', query_mask.numpy(), i = self.num_devices)  # (num_devices, batch_size, query_length)
      ground_truth = einshape('(ij)...->ij...', ground_truth.numpy(), i = self.num_devices)  # (num_devices, batch_size, query_length, value_dim)

    if return_raw:
      return raw, equation, prompt, mask, query, query_mask, ground_truth
    else:
      return equation, prompt, mask, query, query_mask, ground_truth
    
  def pretty_print(self, equation, prompt, mask, query, query_mask, ground_truth):
    pprint(equation)
    print('prompt size: {}'.format(tree.tree_map(lambda x: x.shape, prompt)), flush=True)
    print('mask size: {}'.format(tree.tree_map(lambda x: x.shape, mask)), flush=True)
    print('query size: {}'.format(tree.tree_map(lambda x: x.shape, query)), flush=True)
    print('query_mask size: {}'.format(tree.tree_map(lambda x: x.shape, query_mask)), flush=True)
    print('ground_truth size: {}'.format(tree.tree_map(lambda x: x.shape, ground_truth)), flush=True)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
    dataset = get_tf_dataset(seed = 0, file_names = ["data0407/train*"], 
                            batch_size = 32, num_epochs = 200, shuffle_buffer_size = 10000, 
                            demo_num_min = 4,
                            demo_num_max = 5,
                            cond_len_min = 50,
                            cond_len_max = 50,
                            qoi_len_min = 50,
                            qoi_len_max = 50,
                            k_dim = 2,
                            v_dim = 1,
                            preprocess = 'id')
    iterator = iter(dataset)
    for _ in range(10):
      print('======================================')
      equation, prompt, mask, query, query_mask, ground_truth = iterator.next()
      print(np.concatenate([prompt[0,...], 10*mask[0,...][:,None]], axis = 1))
      print(np.concatenate([query[0,...], ground_truth[0,...], 10*query_mask[0,...][:,None]], axis = 1))
      print([s.decode('utf-8') for s in equation.numpy()])
