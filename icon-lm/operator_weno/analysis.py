import torch
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from pprint import pprint

import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from functools import partial
import haiku as hk
import optax
import pytz
from datetime import datetime
import pickle
from absl import app, flags, logging
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape
import gc
import glob
from pprint import pprint

import sys
sys.path.append('../')
import utils
import plot

from dataloader import Data
from models_utils_pytorch import InputData, inputdata_transform
from runner_jax import Runner_lm
from runner_deepo_torch import Runner as Runner_Operator

import jax
gpu_num = jax.local_device_count()

def append_dict_list(dict, key, value):
  if key not in dict:
    dict[key] = []
  dict[key].append(value)
  return dict

def build_data_dummy(bs_tuple, demo_num, backend):
  # e.g., bs_tuple = (2,8), demo_num = 5
  if backend == "jax":
    data = Data(input_id = None,
                embedding_raw = None, 
                embedding_pool = None, 
                embedding_mask = None, 
                demo_cond_k = np.zeros((*bs_tuple,demo_num,100,3)),
                demo_cond_v = np.zeros((*bs_tuple,demo_num,100,1)),
                demo_cond_mask = np.ones((*bs_tuple,demo_num,100), dtype=bool), 
                demo_qoi_k = np.zeros((*bs_tuple,demo_num,100,3)),
                demo_qoi_v = np.zeros((*bs_tuple,demo_num,100,1)),
                demo_qoi_mask = np.ones((*bs_tuple,demo_num,100), dtype=bool),
                quest_cond_k = np.zeros((*bs_tuple,1,100,3)),
                quest_cond_v = np.zeros((*bs_tuple,1,100,1)),
                quest_cond_mask = np.ones((*bs_tuple,1,100), dtype=bool),
                quest_qoi_k = np.zeros((*bs_tuple,1,100,3)),
                quest_qoi_mask = np.ones((*bs_tuple,1,100), dtype=bool), 
                )
  elif backend == "torch":
    data = InputData(input_id = None, 
                embedding_mask = None,
                demo_cond_k = np.zeros((*bs_tuple,demo_num,100,3), dtype = np.float32),
                demo_cond_v = np.zeros((*bs_tuple,demo_num,100,1), dtype = np.float32),
                demo_cond_mask = np.ones((*bs_tuple,demo_num,100), dtype=bool), 
                demo_qoi_k = np.zeros((*bs_tuple,demo_num,100,3), dtype = np.float32),
                demo_qoi_v = np.zeros((*bs_tuple,demo_num,100,1), dtype = np.float32),
                demo_qoi_mask = np.ones((*bs_tuple,demo_num,100), dtype=bool),
                quest_cond_k = np.zeros((*bs_tuple,1,100,3), dtype = np.float32),
                quest_cond_v = np.zeros((*bs_tuple,1,100,1), dtype = np.float32),
                quest_cond_mask = np.ones((*bs_tuple,1,100), dtype=bool),
                quest_qoi_k = np.zeros((*bs_tuple,1,100,3), dtype = np.float32),
                quest_qoi_mask = np.ones((*bs_tuple,1,100), dtype=bool), 
                )
  return data

def build_data_from_raw(u1s, u2s, x1s, x2s, bs_tuple, demo_num, backend):
  # (bs, 101, 100, 1) (bs, 101, 100, 1) (bs, 101, 100, 1) (bs, 101, 100, 1)
  # the last pair is for testing, the first 100 pairs are for training/in-context examples

  assert u1s.shape == u2s.shape == x1s.shape == x2s.shape
  demo_cond_k = x1s[:,:demo_num,:,:].reshape((*bs_tuple, demo_num, 100, 1)) # *bs_tuple, demo_num, 100, 1)
  demo_cond_v = u1s[:,:demo_num,:,:].reshape((*bs_tuple, demo_num, 100, 1)) # (*bs_tuple, demo_num, 100, 1)
  demo_qoi_k = x2s[:,:demo_num,:,:].reshape((*bs_tuple, demo_num, 100, 1)) # (*bs_tuple, demo_num, 100, 1)
  demo_qoi_v = u2s[:,:demo_num,:,:].reshape((*bs_tuple, demo_num, 100, 1)) # (*bs_tuple, demo_num, 100, 1)
  quest_cond_k = x1s[:,-1:,:,:].reshape((*bs_tuple, 1, 100, 1)) # (*bs_tuple, 1, 100, 1)
  quest_cond_v = u1s[:,-1:,:,:].reshape((*bs_tuple, 1, 100, 1)) # (*bs_tuple, 1, 100, 1)
  quest_qoi_k = x2s[:,-1:,:,:].reshape((*bs_tuple, 1, 100, 1)) # (*bs_tuple, 1, 100, 1)
  quest_qoi_v = u2s[:,-1:,:,:].reshape((*bs_tuple, 1, 100, 1)) # (*bs_tuple, 1, 100, 1)

  # padd zero to k
  demo_cond_k = np.concatenate([demo_cond_k*0, demo_cond_k*0, demo_cond_k], axis=-1) # (*bs_tuple, demo_num, 100, 3)
  demo_qoi_k = np.concatenate([demo_qoi_k*0, demo_qoi_k*0, demo_qoi_k], axis=-1) # (*bs_tuple, demo_num, 100, 3)
  quest_cond_k = np.concatenate([quest_cond_k*0, quest_cond_k*0, quest_cond_k], axis=-1) # (*bs_tuple, 1, 100, 3)
  quest_qoi_k = np.concatenate([quest_qoi_k*0, quest_qoi_k*0, quest_qoi_k], axis=-1) # (*bs_tuple, 1, 100, 3)

  # the original data might be jax array
  demo_cond_v = np.array(demo_cond_v) # (*bs_tuple, demo_num, 100, 1)
  demo_qoi_v = np.array(demo_qoi_v) # (*bs_tuple, demo_num, 100, 1)
  quest_cond_v = np.array(quest_cond_v) # (*bs_tuple, 1, 100, 1)
  quest_qoi_v = np.array(quest_qoi_v) # (*bs_tuple, 1, 100, 1)

  if backend == "jax":
    data = Data(input_id = None,
                embedding_raw = None, 
                embedding_pool = None, 
                embedding_mask = None, 
                demo_cond_k = demo_cond_k,
                demo_cond_v = demo_cond_v,
                demo_cond_mask = np.ones((*bs_tuple, demo_num, 100), dtype=bool),
                demo_qoi_k = demo_qoi_k,
                demo_qoi_v = demo_qoi_v,
                demo_qoi_mask = np.ones((*bs_tuple, demo_num, 100), dtype=bool),
                quest_cond_k = quest_cond_k,
                quest_cond_v = quest_cond_v,
                quest_cond_mask = np.ones((*bs_tuple, 1, 100), dtype=bool),
                quest_qoi_k = quest_qoi_k,
                quest_qoi_mask = np.ones((*bs_tuple, 1, 100), dtype=bool),
                )
  elif backend == "torch":
    data = InputData(input_id = None,
                embedding_mask = None,
                demo_cond_k = demo_cond_k,
                demo_cond_v = demo_cond_v,
                demo_cond_mask = np.ones((*bs_tuple, demo_num, 100), dtype=bool),
                demo_qoi_k = demo_qoi_k,
                demo_qoi_v = demo_qoi_v,
                demo_qoi_mask = np.ones((*bs_tuple, demo_num, 100), dtype=bool),
                quest_cond_k = quest_cond_k,
                quest_cond_v = quest_cond_v,
                quest_cond_mask = np.ones((*bs_tuple, 1, 100), dtype=bool),
                quest_qoi_k = quest_qoi_k,
                quest_qoi_mask = np.ones((*bs_tuple, 1, 100), dtype=bool),
                )
  label = quest_qoi_v[...,0,:,:] # remove the num axis
  return data, label


def get_icon_runner(model_config_filename, seed):

  model_config = utils.load_json("../config_model/" + model_config_filename)
  model_config['caption_len'] = 0

  optimizer = optax.adamw(0.0001) # dummy optimizer

  data = build_data_dummy(bs_tuple = (gpu_num,8), demo_num = 5, backend = "jax") # 2 GPUs, 8 bs per GPU, 5 demo examples
  runner = Runner_lm(seed = seed,
                  model = 'icon_lm',
                  data = data,
                  model_config = model_config,
                  optimizer = optimizer,
                  trainable_mode = 'all',
                  loss_mode = ['nocap'],
                  )
  out = runner.get_pred(data = data, with_caption = False)
  print("---------------------- model construction -----------------------")
  print("input shape", tree.tree_map(lambda x: x.shape, data))
  print("output shape", out.shape)
  print("---------------------- model construction end ----------------------")
  
  return runner


def get_operator_runner(model_config_filename, model, seed, opt_config = None):

  model_config = utils.load_json("../config_model/" + model_config_filename)
  if opt_config is None:
    opt_config = {'peak_lr': FLAGS.lr,
                  'end_lr': FLAGS.lr,
                  'warmup_steps': 1, # the first step is warmup, i.e. lr = 0
                  'decay_steps': FLAGS.tune_steps,
                  'gnorm_clip': 1,
                  'weight_decay': 0.0001,
                  }

  runner = Runner_Operator(data = None, model_config = model_config, opt_config = opt_config, 
                          model_name = model, 
                          pretrained = False, trainable_mode = None, # useless
                          loss_mode = ['demo'], # only fine-tune with demo
                          )

  data = build_data_dummy(bs_tuple = (8,), demo_num = 20, backend = "torch") # bs = 8, 20 demo examples, no gpu dim
  out = runner.get_pred(data = data, with_caption = False)
  print("---------------------- model construction -----------------------")
  print("input shape", inputdata_transform(lambda x: x.shape, data))
  print("output shape", out.shape)
  print("---------------------- model construction end ----------------------")

  return runner


def make_plot(label, out, title):
  plt.figure(figsize=(20,5))
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.plot(label[0,i,:,0], 'k-')
    plt.plot(out[0,i,:,0], 'r--')
  plt.savefig("{}.png".format(title))

def test_icon(model_path, model_step, data_path):
  runner = get_icon_runner(model_config_filename = 'model_lm_config.json', seed = 1)
  runner.restore(model_path, model_step, restore_opt_state = False)
  #load data from pickle
  with open(data_path, 'rb') as file:
    raw_data = pickle.load(file)

  all_u1s, all_u2s, all_x1s, all_x2s, all_params = raw_data
  print(all_u1s[0].shape, all_u2s[0].shape, all_x1s[0].shape, all_x2s[0].shape, all_params[0])
  # there is only on element in each list, since only one equation
  # (100, 101, 100, 1) (100, 101, 100, 1) (100, 101, 100, 1) (100, 101, 100, 1)
  # 100 groups, 101 pairs in each group
  # the last pair is for testing, the first 100 pairs are for training/in-context examples
  data, label = build_data_from_raw(all_u1s[0], all_u2s[0], all_x1s[0], all_x2s[0], 
                                    bs_tuple = (gpu_num,100//gpu_num), demo_num = 5, backend = "jax")
  print("test input shape", tree.tree_map(lambda x: x.shape, data))
  out = runner.get_pred(data = data, with_caption = False)
  print("test output shape", out.shape, "label shape", label.shape)
  error = np.mean(np.abs(out - label))
  print("error", error)
  return label, out


def test_operator_untune(model, model_path, model_step, data_path):
  if model == 'deepo':
    runner = get_operator_runner(model_config_filename = 'model_deepo_weno_config.json', model = 'deepo', seed = 1)
  elif model == 'fno':
    runner = get_operator_runner(model_config_filename = 'model_fno_weno_config.json', model = 'fno', seed = 1)
  
  runner.restore(model_path, model_step, restore_opt_state = False)
  #load data from pickle
  with open(data_path, 'rb') as file:
    raw_data = pickle.load(file)

  all_u1s, all_u2s, all_x1s, all_x2s, all_params = raw_data
  print(all_u1s[0].shape, all_u2s[0].shape, all_x1s[0].shape, all_x2s[0].shape, all_params[0])
  data, label = build_data_from_raw(all_u1s[0], all_u2s[0], all_x1s[0], all_x2s[0], 
                                    bs_tuple = (100,), demo_num = 5, backend = "torch")
  print("test input shape", inputdata_transform(lambda x: x.shape, data))
  out = runner.get_pred(data = data, with_caption = False)
  print("test output shape", out.shape, "label shape", label.shape)
  error = np.mean(np.abs(out - label))
  print("error", error)
  return label, out


def test_operator_tune(model, model_path, model_step, data_path, demo_num):
  if model == 'deepo':
    runner = get_operator_runner(model_config_filename = 'model_deepo_weno_config.json', model = 'deepo', seed = 1)
  elif model == 'fno':
    runner = get_operator_runner(model_config_filename = 'model_fno_weno_config.json', model = 'fno', seed = 1)
    
  #load data from pickle
  with open(data_path, 'rb') as file:
    raw_data = pickle.load(file)

  all_u1s, all_u2s, all_x1s, all_x2s, all_params = raw_data
  print(all_u1s[0].shape, all_u2s[0].shape, all_x1s[0].shape, all_x2s[0].shape, all_params[0])
  all_gt = []
  all_pred = []
  all_pred_tune = []
  for bid in range(len(all_u1s[0])):
    data, label = build_data_from_raw(all_u1s[0][bid:bid+1,...], all_u2s[0][bid:bid+1,...], 
                                      all_x1s[0][bid:bid+1,...], all_x2s[0][bid:bid+1,...], 
                                    bs_tuple = (1,), demo_num = demo_num, backend = "torch")
    label = label[:,None,...] # add the num axis
    runner.restore(model_path, model_step, restore_opt_state = False)
    runner.reset_optimizer()
    init_error, pred = runner.get_error(data, label, with_caption = False, return_pred = True) # (bs, query_len, 1)
    print("bid", bid, "error", np.mean(init_error), flush=True)
    # fine-tune the model
    all_pred_tune.append([])
    all_pred_tune[-1].append(pred)
    for n in range(1, FLAGS.tune_steps+1):
        runner.iter(data, 0) # should not use label
        if n % 100 == 0 or (n < FLAGS.tune_steps/10 and n % 10 == 0):
          print('bid', bid, 'n', n, end = " ")
          loss = runner.get_loss(data, 0) # should not use label
          print("loss", loss, end = " ")
          error, pred = runner.get_error(data, label, with_caption = False, return_pred = True) # (bs, query_len, 1)
          print("error", np.mean(error), np.mean(init_error), flush=True)
          all_pred_tune[-1].append(pred)

    all_gt.append(label)
    all_pred.append(pred)

  all_gt = np.concatenate(all_gt, axis=0)
  all_pred = np.concatenate(all_pred, axis=0)
  print("all error", np.mean(np.abs(all_gt - all_pred)))
  all_pred_tune = np.array(all_pred_tune)
  return all_gt, all_pred, all_pred_tune

def main(argv):

  # PART 1: test icon
  for data_path in [
                    "fix_0.30_0.30_0.30.pkl", 
                    "fix_0.25_0.25_0.25.pkl", 
                    "fix_0.21_0.21_0.21.pkl", 
                    "fix_0.20_0.20_0.20.pkl",
                    "fixdirichlet_0.20_0.20_0.20.pkl" # dirichlet boundary condition
                    ]:
    label, out = test_icon(model_path = '/home/shared/icon/save/user/ckpts/icon_weno/20231209-222440', 
                  model_step = 1000000, 
                  data_path = data_path)
    np.savez("icon_data_{}_demonum_{}.npz".format(data_path, 5), label = label, out = out)
  gc.collect()

  # PART2: test operator on training operator
  for data_path in ["fix_0.30_0.30_0.30.pkl", 
                    "fix_0.25_0.25_0.25.pkl", 
                    "fix_0.21_0.21_0.21.pkl", 
                    "fix_0.20_0.20_0.20.pkl"]:
    label, out = test_operator_untune(model = 'deepo',
                  model_path = '/home/shared/icon/save/user/ckpts/weno_deepo_pretrain/20240604-222907',
                  model_step = 100000,
                  data_path = data_path)
    np.savez("notune_deepo_data_{}.npz".format(data_path), label = label, out = out)
    gc.collect()

    # NOTE: FNO can only run with a single GPU...
    label, out = test_operator_untune(model = 'fno',
                  model_path = '/home/shared/icon/save/user/ckpts/weno_fno_pretrain/20240604-222915',
                  model_step = 100000,
                  data_path = data_path)
    np.savez("notune_fno_data_{}.npz".format(data_path), label = label, out = out)
    gc.collect()

  # PART3: fine tune operator learning models
  for data_path in [
                    "fix_0.30_0.30_0.30.pkl", 
                    "fix_0.25_0.25_0.25.pkl", 
                    "fix_0.21_0.21_0.21.pkl", 
                    "fix_0.20_0.20_0.20.pkl"
                    ]:
    for demo_num in [10]:
      label, out, pred_tune = test_operator_tune(model = 'fno',
                    model_path = '/home/shared/icon/save/user/ckpts/weno_fno_pretrain/20240604-222915',
                    model_step = 100000,
                    data_path = data_path,
                    demo_num = demo_num)
      np.savez("tune_all_fno_data_{}_demonum_{}.npz".format(data_path, demo_num), label = label, out = out, pred_tune = pred_tune)
      gc.collect()

      label, out, pred_tune = test_operator_tune(model = 'deepo',
                    model_path = '/home/shared/icon/save/user/ckpts/weno_deepo_pretrain/20240604-222907',
                    model_step = 100000,
                    data_path = data_path,
                    demo_num = demo_num)
      np.savez("tune_all_deepo_data_{}_demonum_{}.npz".format(data_path, demo_num), label = label, out = out, pred_tune = pred_tune)
      gc.collect()
      
if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('tune_steps', 1000, 'tune steps')
  flags.DEFINE_float('lr', 0.00001, 'learning rate')
  app.run(main)
