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
import analysis as ans


def test_icon(model_path, model_step, data_path, bs, repeat):
  runner = ans.get_icon_runner(model_config_filename = 'model_lm_config.json', seed = 1)
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
  data, label = ans.build_data_from_raw(all_u1s[0][0:bs,...], all_u2s[0][0:bs,...], all_x1s[0][0:bs,...], all_x2s[0][0:bs,...], 
                                    bs_tuple = (gpu_num,bs//gpu_num), demo_num = 5, backend = "jax")
  print("test input shape", tree.tree_map(lambda x: x.shape, data))
  
  # warm up for JIT
  for _ in range(10):
    out = runner.get_pred(data = data, with_caption = False)
  # timing
  utils.timer.tic("icon")
  for _ in range(repeat):
    out = runner.get_pred(data = data, with_caption = False)
  utils.timer.toc("icon") 
  print("model = {}, repeat = {}, steps = {}, time = {:.3f}".format("icon", repeat, 1, utils.timer.get_time("icon")))
  print("average time for each forward of {}: {:.3f}".format("icon", utils.timer.get_time("icon") / repeat))
        
  out = runner.get_pred(data = data, with_caption = False)
  print("test output shape", out.shape, "label shape", label.shape)
  error = np.mean(np.abs(out - label))
  print("error", error)
  return label, out


def test_operator_untune(model, model_path, model_step, data_path, bs, repeat):
  # dummy opt_config
  opt_config = {'peak_lr': FLAGS.lr,
                'end_lr': FLAGS.lr,
                'warmup_steps': 1, # the first step is warmup, i.e. lr = 0
                'decay_steps': FLAGS.tune_steps,
                'gnorm_clip': 1,
                'weight_decay': 0.0001,
                }
  runner = ans.get_operator_runner(model_config_filename = f'model_{model}_weno_config.json', model = model, seed = 1, opt_config = opt_config)
  
  runner.restore(model_path, model_step, restore_opt_state = False)

  #load data from pickle
  with open(data_path, 'rb') as file:
    raw_data = pickle.load(file)

  all_u1s, all_u2s, all_x1s, all_x2s, all_params = raw_data
  print(all_u1s[0].shape, all_u2s[0].shape, all_x1s[0].shape, all_x2s[0].shape, all_params[0])
  data, label = ans.build_data_from_raw(all_u1s[0][0:bs,...], all_u2s[0][0:bs,...], all_x1s[0][0:bs,...], all_x2s[0][0:bs,...], 
                                    bs_tuple = (bs,), demo_num = 5, backend = "torch")
  print("test input shape", inputdata_transform(lambda x: x.shape, data))
  
  # warm up for JIT
  for _ in range(10):
    out = runner.get_pred(data = data, with_caption = False)
  # timing
  utils.timer.tic(model+"_untune")
  for _ in range(repeat):
    out = runner.get_pred(data = data, with_caption = False)
  utils.timer.toc(model+"_untune")
  print("model = {}, repeat = {}, steps = {}, time = {:.3f}".format(model, repeat, 1, utils.timer.get_time(model+"_untune")))
  print("average time for each forward of {}: {:.3f}".format(model, utils.timer.get_time(model+"_untune") / repeat))
  
  out = runner.get_pred(data = data, with_caption = False)
  print("test output shape", out.shape, "label shape", label.shape)
  error = np.mean(np.abs(out - label))
  print("error", error)
  return label, out



def test_operator_tune(model, model_path, model_step, data_path, demo_num, repeat):

  opt_config = {'peak_lr': FLAGS.lr,
                'end_lr': FLAGS.lr,
                'warmup_steps': 1, # the first step is warmup, i.e. lr = 0
                'decay_steps': FLAGS.tune_steps,
                'gnorm_clip': 1,
                'weight_decay': 0.0001,
                }
    
  runner = ans.get_operator_runner(model_config_filename = f'model_{model}_weno_config.json', model = model, seed = 1, opt_config = opt_config)
    
  #load data from pickle
  with open(data_path, 'rb') as file:
    raw_data = pickle.load(file)

  all_u1s, all_u2s, all_x1s, all_x2s, all_params = raw_data
  print(all_u1s[0].shape, all_u2s[0].shape, all_x1s[0].shape, all_x2s[0].shape, all_params[0])
  all_gt = []
  all_pred = []
  bid = 0
  data, label = ans.build_data_from_raw(all_u1s[0][bid:bid+1,...], all_u2s[0][bid:bid+1,...], 
                                    all_x1s[0][bid:bid+1,...], all_x2s[0][bid:bid+1,...], 
                                  bs_tuple = (1,), demo_num = demo_num, backend = "torch")
  label = label[:,None,...] # add the num axis
  runner.restore(model_path, model_step, restore_opt_state = False)
  runner.reset_optimizer()
  init_error, pred = runner.get_error(data, label, with_caption = False, return_pred = True) # (bs, query_len, 1)
  print("bid", bid, "error", np.mean(init_error), flush=True)
  # fine-tune the model
  
  # warm up for JIT
  for _ in range(100):
    runner.iter(data, 0)
  
  # timing
  times = []
  for r in range(repeat):
    runner.restore(model_path, model_step, restore_opt_state = False)
    runner.reset_optimizer()
    print("repeat", r)
    print("data shape", inputdata_transform(lambda x: x.shape, data))
    utils.timer.tic(model+"_tune")
    for n in range(FLAGS.tune_steps):
        runner.iter(data, 0) # should not use label
    utils.timer.toc(model+"_tune")
    times.append(utils.timer.get_time(model+"_tune"))

  print("times", times)
  average_time = np.mean(times)
  print("model = {}, examples = {}, repeat = {}, steps = {}, time = {:.3f}".format(model, demo_num, repeat, FLAGS.tune_steps, average_time))
  print("average time for tuning {}: {:.3f}".format(model, average_time))

def main(argv):

  # Add CUDA_VISIBLE_DEVICES="0" to the command line to ensure single GPU.
  data_path = "fix_0.30_0.30_0.30.pkl"

  # PART 1: test icon
  test_icon(model_path = '/home/shared/icon/save/user/ckpts/icon_weno/20231209-222440', 
                  model_step = 1000000, 
                  data_path = data_path,
                  bs = 1, repeat = 100)
  gc.collect()
  print("====================icon test done=================")

  # PART 2: test operator
  test_operator_untune(model = 'deepo',
                  model_path = '/home/shared/icon/save/user/ckpts/weno_deepo_pretrain/20240604-222907',
                  model_step = 100000,
                  data_path = data_path,
                  bs = 1, repeat = 100)
  gc.collect()
  print("====================operator untune test done=================")

  test_operator_untune(model = 'fno',
                  model_path = '/home/shared/icon/save/user/ckpts/weno_fno_pretrain/20240604-222915',
                  model_step = 100000,
                  data_path = data_path,
                  bs = 1, repeat = 100)
  gc.collect()
  print("====================operator untune test done=================")

  # PART 3: test operator tune
  for demo_num in [5,100]:
    test_operator_tune(model = 'deepo',
                    model_path = '/home/shared/icon/save/user/ckpts/weno_deepo_pretrain/20240604-222907',
                    model_step = 100000,
                    data_path = data_path,
                    demo_num = demo_num, repeat = 10)
    gc.collect()
    print("====================operator tune test done=================")

    test_operator_tune(model = 'fno',
                    model_path = '/home/shared/icon/save/user/ckpts/weno_fno_pretrain/20240604-222915',
                    model_step = 100000,
                    data_path = data_path,
                    demo_num = demo_num, repeat = 10)
    
    gc.collect()
    print("====================operator tune test done=================")


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('tune_steps', 1000, 'tune steps')
  flags.DEFINE_float('lr', 0.00001, 'learning rate')
  app.run(main)

