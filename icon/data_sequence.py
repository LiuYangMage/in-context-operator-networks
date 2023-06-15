import numpy as np
import tensorflow as tf

tf_rng_seq = tf.random.Generator.from_seed(1234)

@tf.function
def select_kv(key, val, len_full, len_select, select_method):
    '''
    select some k-v pairs from the full set of k-v pairs, and append them to the list of k-v pair lists
    if len_select > len_full, then select all, and pad with 0.
    @ param:
        seed: int
        key: 2D array, [len, kdim]  (len >= len_full)
        val: 2D array, [len, vdim]
        key_list, val_list: lists of key and val that the new k-v need to be appended to
        len_full, len_select: int
        select_method: 'random' or 'even' or 'first'
    @ return:
        key_list: the updated list of 2D arrays [len_select, kdim]
        val_list: the updated list of 2D arrays [len_select, vdim]
    '''
    if len_select > len_full:
        key = tf.pad(key, [[0, len_select - len_full], [0, 0]])
        val = tf.pad(val, [[0, len_select - len_full], [0, 0]])
    # sometimes we need shuffle even if len_select == len_full
    # elif len_select == len_full: 
    #     pass
    else: # len_select < len_full
        if select_method == 'random':
            seed = tf_rng_seq.make_seeds(1)[:,0]
            index = tf.random.experimental.stateless_shuffle(tf.range(len_full), seed = seed)[0:len_select]
        elif select_method == 'even':
            delta = (len_full - 1) // (len_select - 1)
            index = tf.range(0, len_select) * delta
        elif select_method == 'first':
            index = tf.range(0, len_select)
        key = tf.gather(key, index, axis = 0)
        val = tf.gather(val, index, axis = 0)
    return key, val


def apply_random_demo_num_in_use(demo_num, config, demo_cond_mask_list, demo_qoi_mask_list):
  '''
  randomly select the number of demos to be used in the current prompt
  '''
  demo_num_in_use = tf_rng_seq.uniform(shape = (), minval = config['demo_num_begin'], maxval = config['demo_num_end'], dtype = tf.int32)
  demo_in_use_mask = tf.pad(tf.ones((demo_num_in_use), dtype = tf.int32), [[0, demo_num - demo_num_in_use]])
  new_demo_cond_mask_list = []
  new_demo_qoi_mask_list = []
  for i in range(demo_num):
    new_demo_cond_mask_list.append(demo_in_use_mask[i] * demo_cond_mask_list[i])
    new_demo_qoi_mask_list.append(demo_in_use_mask[i] * demo_qoi_mask_list[i])
  return new_demo_cond_mask_list, new_demo_qoi_mask_list

def apply_cond_qoi_len_in_use(demo_num, cond_len, qoi_len, config,
                              cond_len_in_use = None, qoi_len_in_use = None,
                              demo_cond_mask_list = None, demo_qoi_mask_list = None, quest_cond_mask = None, quest_qoi_mask = None):
  '''
  apply cond_len_in_use and qoi_len_in_use to the original masks
  '''
  if demo_cond_mask_list is None:
    demo_cond_mask_list = [1 for _ in range(demo_num)]
  if demo_qoi_mask_list is None:
    demo_qoi_mask_list = [1 for _ in range(demo_num)]
  if quest_cond_mask is None:
    quest_cond_mask = [1]
  if quest_qoi_mask is None:
    quest_qoi_mask = [1]

  if cond_len_in_use is None:
      cond_len_in_use = tf_rng_seq.uniform(shape = (demo_num + 1,),
                                minval = config['cond_len_in_use_begin'],
                                maxval = config['cond_len_in_use_end'], dtype = tf.int32)
  if qoi_len_in_use is None:
      qoi_len_in_use = tf_rng_seq.uniform(shape = (demo_num + 1,),
                                minval = config['qoi_len_in_use_begin'],
                                maxval = config['qoi_len_in_use_end'], dtype = tf.int32)
      
  new_demo_cond_mask_list = []
  new_demo_qoi_mask_list = []
  for i in range(demo_num):
    cond_len_in_use_i = cond_len_in_use[i]
    qoi_len_in_use_i = qoi_len_in_use[i]
    cond_len_mask_i = tf.pad(tf.ones((cond_len_in_use_i,), dtype = tf.int32), [[0, cond_len - cond_len_in_use_i]])
    qoi_len_mask_i = tf.pad(tf.ones((qoi_len_in_use_i,), dtype = tf.int32), [[0, qoi_len - qoi_len_in_use_i]])
    new_demo_cond_mask_list.append(cond_len_mask_i * demo_cond_mask_list[i])
    new_demo_qoi_mask_list.append(qoi_len_mask_i * demo_qoi_mask_list[i])
  
  cond_len_in_use_quest = cond_len_in_use[-1]
  qoi_len_in_use_quest = qoi_len_in_use[-1]
  cond_len_mask_quest = tf.pad(tf.ones((cond_len_in_use_quest,), dtype = tf.int32), [[0, cond_len - cond_len_in_use_quest]])
  qoi_len_mask_quest = tf.pad(tf.ones((qoi_len_in_use_quest,), dtype = tf.int32), [[0, qoi_len - qoi_len_in_use_quest]])
  new_quest_cond_mask = [cond_len_mask_quest * quest_cond_mask[0]]
  new_quest_qoi_mask = [qoi_len_mask_quest * quest_qoi_mask[0]]

  return new_demo_cond_mask_list, new_demo_qoi_mask_list, new_quest_cond_mask, new_quest_qoi_mask


def build_function_kv(demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                      quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                      demo_num, cond_len, qoi_len, cond_full_len, qoi_full_len, config):
  '''
  apply select_kv to all demos and quest
  select cond_len tokens in the range of [0, cond_full_len], 
  if cond_len > cond_full_len, pad with zero
  similarly for qoi
  '''
  demo_cond_k_list = []
  demo_cond_v_list = []
  demo_qoi_k_list = []
  demo_qoi_v_list = []
  for i in range(demo_num):
    this_demo_cond_k, this_demo_cond_v = select_kv(demo_cond_k[i,...], demo_cond_v[i,...], cond_full_len, cond_len, config['select_cond_ind'])
    this_demo_qoi_k, this_demo_qoi_v = select_kv(demo_qoi_k[i,...], demo_qoi_v[i,...], qoi_full_len, qoi_len, config['select_qoi_ind'])
    demo_cond_k_list.append(this_demo_cond_k)
    demo_cond_v_list.append(this_demo_cond_v)
    demo_qoi_k_list.append(this_demo_qoi_k)
    demo_qoi_v_list.append(this_demo_qoi_v)

  this_quest_cond_k, this_quest_cond_v = select_kv(quest_cond_k[0,...], quest_cond_v[0,...], cond_full_len, cond_len, config['select_cond_ind'])
  this_quest_qoi_k, this_quest_qoi_v = select_kv(quest_qoi_k[0,...], quest_qoi_v[0,...], qoi_full_len, qoi_len, config['select_qoi_ind'])
  quest_cond_k = [this_quest_cond_k]
  quest_cond_v = [this_quest_cond_v]
  quest_qoi_k = [this_quest_qoi_k]
  quest_qoi_v = [this_quest_qoi_v]
  
  return demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v


def build_ode_forward(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                          quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                          demo_num, cond_len, qoi_len, config):
  '''
  config['select_cond_ind'] and config['select_qoi_ind'] must be "even" or "first"
  2 <= qoi_len_in_use_begin < qoi_len_in_use_end <= qoi_len + 1
  cond_len_in_use = qoi_len_in_use - 1, then adding the last one as in use (initial condition)
  case 1, qoi_len_in_use == 2, cond_len_in_use = 1: 1 control token, 1 initial condition token
  case 2, qoi_len_in_use == qoi_len, cond_len_in_use = qoi_len-1: qoi_len-1 control token, 1 initial condition token
  '''
  cond_full_len = tf.shape(demo_cond_k)[1]
  qoi_full_len = tf.shape(demo_qoi_k)[1]
  demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
    quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v = build_function_kv(demo_cond_k, demo_cond_v, 
                      demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                      demo_num, cond_len, qoi_len, cond_full_len, qoi_full_len, config)

  qoi_len_in_use = tf_rng_seq.uniform(shape = (demo_num + 1,),
                                minval = config['qoi_len_in_use_begin'],
                                maxval = config['qoi_len_in_use_end'], dtype = tf.int32)
  cond_len_in_use = qoi_len_in_use - 1
  
  demo_cond_mask_list = []
  demo_qoi_mask_list = []
  for i in range(demo_num):
    cond_len_in_use_i = cond_len_in_use[i]
    qoi_len_in_use_i = qoi_len_in_use[i]
    demo_cond_mask_list.append(tf.concat([tf.ones((cond_len_in_use_i,), dtype = tf.int32),
                                          tf.zeros((cond_len-1-cond_len_in_use_i), dtype = tf.int32), 
                                          tf.ones((1), dtype = tf.int32)], axis = 0)) # keep the last one
    demo_qoi_mask_list.append(tf.pad(tf.ones((qoi_len_in_use_i,), dtype = tf.int32),[[0, qoi_len-qoi_len_in_use_i]]))

  cond_len_in_use_quest = cond_len_in_use[-1]
  qoi_len_in_use_quest = qoi_len_in_use[-1]
  quest_cond_mask = [tf.concat([tf.ones((cond_len_in_use_quest,), dtype = tf.int32),
                    tf.zeros((cond_len-1-cond_len_in_use_quest), dtype = tf.int32), 
                    tf.ones((1), dtype = tf.int32)], axis = 0)]
  quest_qoi_mask = [tf.pad(tf.ones((qoi_len_in_use_quest,), dtype = tf.int32), [[0, qoi_len-qoi_len_in_use_quest]])]

  demo_cond_mask_list, demo_qoi_mask_list = apply_random_demo_num_in_use(demo_num, config, demo_cond_mask_list, demo_qoi_mask_list)
  return equation, demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
                  demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask, quest_qoi_mask


def build_ode_inverse(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                          quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                          demo_num, cond_len, qoi_len, config):
  '''
  config['select_cond_ind'] and config['select_qoi_ind'] must be "even" or "first"
  1 <= qoi_len_in_use_begin < qoi_len_in_use_end <= qoi_len
  cond_len_in_use = qoi_len_in_use + 1
  case 1, qoi_len_in_use == 1, cond_len_in_use = 2
  case 2, qoi_len_in_use == qoi_len - 1, cond_len_in_use = qoi_len
  '''
  cond_full_len = tf.shape(demo_cond_k)[1]
  qoi_full_len = tf.shape(demo_qoi_k)[1]
  demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
    quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v = build_function_kv(demo_cond_k, demo_cond_v, 
                      demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                      demo_num, cond_len, qoi_len, cond_full_len, qoi_full_len, config)

  qoi_len_in_use = tf_rng_seq.uniform(shape = (demo_num + 1,),
                                minval = config['qoi_len_in_use_begin'],
                                maxval = config['qoi_len_in_use_end'], dtype = tf.int32)
  cond_len_in_use = qoi_len_in_use + 1

  demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask, quest_qoi_mask = \
                      apply_cond_qoi_len_in_use(demo_num, cond_len, qoi_len, config, 
                                      cond_len_in_use = cond_len_in_use, qoi_len_in_use = qoi_len_in_use)
  demo_cond_mask_list, demo_qoi_mask_list = apply_random_demo_num_in_use(demo_num, config, demo_cond_mask_list, demo_qoi_mask_list)

  return equation, demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
                  demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask, quest_qoi_mask



def build_others(equation, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v,
                          quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                          demo_num, cond_len, qoi_len, config):
  cond_full_len = tf.shape(demo_cond_k)[1]
  qoi_full_len = tf.shape(demo_qoi_k)[1]
  demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
    quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v = build_function_kv(demo_cond_k, demo_cond_v, 
                      demo_qoi_k, demo_qoi_v, quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, 
                      demo_num, cond_len, qoi_len, cond_full_len, qoi_full_len, config)
  

  demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask, quest_qoi_mask = apply_cond_qoi_len_in_use(demo_num, cond_len, qoi_len, config)
  demo_cond_mask_list, demo_qoi_mask_list = apply_random_demo_num_in_use(demo_num, config, demo_cond_mask_list, demo_qoi_mask_list)

  return equation, demo_cond_k_list, demo_cond_v_list, demo_qoi_k_list, demo_qoi_v_list, \
                  quest_cond_k, quest_cond_v, quest_qoi_k, quest_qoi_v, \
                  demo_cond_mask_list, demo_qoi_mask_list, quest_cond_mask, quest_qoi_mask


build_pde_spatial_forward = build_others
build_pde_spatial_inverse = build_others
build_time_series = build_others
build_mfc_gparam_forward = build_others
build_mfc_rhoparam_forward = build_others
