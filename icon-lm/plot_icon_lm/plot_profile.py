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

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

def run_analysis():

  utils.set_seed(FLAGS.seed)
  from dataloader import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader

  test_data_dirs = FLAGS.test_data_dirs
  test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in FLAGS.test_data_globs]

  print("test_file_names: ", flush=True)
  pprint(test_file_names)

  model_config = utils.load_json("../config_model/" + FLAGS.model_config_filename)
  test_config = utils.load_json("../config_data/" + FLAGS.test_config_filename)

  if 'cap' not in FLAGS.loss_mode:
    model_config['caption_len'] = 0
    test_config['load_list'] = []


  print('==============data config==============', flush = True)
  print("test_config: ", flush=True)
  pprint(test_config)
  print('==============data config end==============', flush = True)

  print('-----------------------model config-----------------------')
  print("model_config: ", flush=True)
  pprint(model_config)
  print('-----------------------model config end-----------------------')


  if FLAGS.backend == 'jax':
    optimizer = optax.adamw(0.0001) # dummy optimizer
    import jax
    data_num_devices = len(jax.devices())
  elif FLAGS.backend == 'torch':
    # dummy optimizer
    opt_config = {'peak_lr': 0.001,
                  'end_lr': 0,
                  'warmup_steps': 10,
                  'decay_steps': 100,
                  'gnorm_clip': 1,
                  'weight_decay': 0.0001,
                  }
    data_num_devices = 0
  else:
    raise ValueError("backend {} not supported".format(FLAGS.backend))

  test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
  test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]

  test_data = DataProvider(seed = FLAGS.seed + 10,
                            config = test_config,
                            file_names = test_file_names,
                            batch_size = FLAGS.batch_size,
                            deterministic = True,
                            drop_remainder = False, 
                            shuffle_dataset = False,
                            num_epochs=1,
                            shuffle_buffer_size=10,
                            num_devices=data_num_devices,
                            real_time = True,
                            caption_home_dir = '../data_preparation',
                          )
  
  raw, equation, caption, data, label = test_data.get_next_data(return_raw=True)
  
  print_eqn_caption(equation, caption, decode = False)
  print("raw: ", tree.tree_map(lambda x: x.shape, raw)) 
  print("data: ", tree.tree_map(lambda x: x.shape, data)) 
  print("label: ", label.shape)
  '''
  raw:  (TensorShape([4, 5, 2500, 2]), 
          TensorShape([4, 5, 2500, 1]), 
          TensorShape([4, 5, 2600, 2]), 
          TensorShape([4, 5, 2600, 1]), 
          TensorShape([4, 1, 2500, 2]), 
          TensorShape([4, 1, 2500, 1]), 
          TensorShape([4, 1, 2600, 2]), 
          TensorShape([4, 1, 2600, 1]))
  data:  Data(input_id=(2, 2), embedding_raw=(2, 2), embedding_pool=(2, 2), embedding_mask=(2, 2), 
              demo_cond_k=(2, 2, 5, 50, 3), demo_cond_v=(2, 2, 5, 50, 1), demo_cond_mask=(2, 2, 5, 50), 
              demo_qoi_k=(2, 2, 5, 50, 3), demo_qoi_v=(2, 2, 5, 50, 1), demo_qoi_mask=(2, 2, 5, 50), 
              quest_cond_k=(2, 2, 1, 50, 3), quest_cond_v=(2, 2, 1, 50, 1), quest_cond_mask=(2, 2, 1, 50), 
              quest_qoi_k=(2, 2, 1, 2600, 3), quest_qoi_mask=(2, 2, 1, 2600))
  label:  (2, 2, 1, 2600, 1)
  '''


  if FLAGS.model in ['icon', 'icon_scale', 'icon_scale_surrogate']:
    from runner_jax import Runner_vanilla
    runner = Runner_vanilla(seed = FLAGS.seed,
                    model = FLAGS.model,
                    data = data,
                    model_config = model_config,
                    optimizer = optimizer,
                    trainable_mode = 'all',
                    )
  elif FLAGS.model in ['icon_lm']:
    from runner_jax import Runner_lm
    runner = Runner_lm(seed = FLAGS.seed,
                    model = FLAGS.model,
                    data = data,
                    model_config = model_config,
                    optimizer = optimizer,
                    trainable_mode = 'all',
                    loss_mode = FLAGS.loss_mode,
                    print_model = False,
                    )
  elif FLAGS.model in ['gpt2']:
    from runner_torch import Runner
    runner = Runner(data, model_config, opt_config = opt_config, 
                    model_name = FLAGS.model, pretrained = True, 
                    trainable_mode = 'all',
                    loss_mode = FLAGS.loss_mode,
                    )
  else:
    raise ValueError("model {} not supported".format(FLAGS.model))
  

  runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
  result_dict = {}

  data = data._replace(demo_cond_k = data.demo_cond_k[...,:FLAGS.demo_num,:,:],
                      demo_cond_v = data.demo_cond_v[...,:FLAGS.demo_num,:,:],
                      demo_cond_mask = data.demo_cond_mask[...,:FLAGS.demo_num,:],
                      demo_qoi_k = data.demo_qoi_k[...,:FLAGS.demo_num,:,:],
                      demo_qoi_v = data.demo_qoi_v[...,:FLAGS.demo_num,:,:],
                      demo_qoi_mask = data.demo_qoi_mask[...,:FLAGS.demo_num,:],
                      )
  
  pred = runner.get_pred(data, with_caption = False)
  print("pred:", pred.shape) #(2, 2, 2600, 1)

  # merge batch and device dimension
  result_dict['raw'] = raw
  result_dict['data'] = tree.tree_map(lambda x : einshape('bi...->(bi)...', x), data)
  result_dict['label'] = einshape('bi...->(bi)...', label)
  result_dict['pred'] = einshape('bi...->(bi)...', pred)
  return result_dict


def plot_2D(result_dict, plot_index):
    
    demo_num = FLAGS.demo_num
    vmin = FLAGS.vmin
    vmax = FLAGS.vmax
    evmin = FLAGS.evmin
    evmax = FLAGS.evmax
    rhocmap = 'GnBu'
    errcmap = 'bwr'
    shading = 'gouraud'
    x_size = 100


    raw = result_dict['raw']
    raw = tree.tree_map(lambda a : einshape("bn(tx)i->bntxi", a, x = x_size), raw)
    raw_demo_cond_k = raw[0][plot_index,...] # [demo_num, 25, 100, 2]
    raw_demo_cond_v = raw[1][plot_index,...] # [demo_num, 25, 100, 1]
    raw_demo_qoi_k = raw[2][plot_index,...] # [demo_num, 26, 100, 2]
    raw_demo_qoi_v = raw[3][plot_index,...] # [demo_num, 26, 100, 1]
    raw_quest_cond_k = raw[4][plot_index,...] # [1, 25, 100, 2]
    raw_quest_cond_v = raw[5][plot_index,...] # [1, 25, 100, 1]
    raw_quest_qoi_k = raw[6][plot_index,...] # [1, 26, 100, 2]
    raw_quest_qoi_v = raw[7][plot_index,...] # [1, 26, 100, 1]

    data = result_dict['data']
    data_demo_cond_k = data.demo_cond_k[plot_index,...] # [demo_num,50,3]
    data_demo_cond_v = data.demo_cond_v[plot_index,...] # [demo_num,50,1]
    data_demo_cond_mask = data.demo_cond_mask[plot_index,...] # [demo_num,50]
    data_demo_qoi_k = data.demo_qoi_k[plot_index,...] # [demo_num,50,3]
    data_demo_qoi_v = data.demo_qoi_v[plot_index,...] # [demo_num,50,1]
    data_demo_qoi_mask = data.demo_qoi_mask[plot_index,...] # [demo_num,50]
    data_quest_cond_k = data.quest_cond_k[plot_index,...] # [1,50,3]
    data_quest_cond_v = data.quest_cond_v[plot_index,...] # [1,50,1]
    data_quest_cond_mask = data.quest_cond_mask[plot_index,...] # [1,50]
    data_quest_qoi_k = data.quest_qoi_k[plot_index,...] # [1,2600,3]
    data_quest_qoi_mask = data.quest_qoi_mask[plot_index,...] # [1,2600]


    pred = einshape("b(tx)i->btxi", result_dict['pred'], x = x_size) # [bs, 26, 100, 1]
    pred = pred[plot_index,...] # [26, 100, 1]

    plt.close('all')
    fig = plt.figure(figsize=(9, 3))
    gs = gridspec.GridSpec(1, 7, width_ratios=[0.1, 0.22, 1, 1, 1, 1, 0.1])  # Create a 1x5 grid of subplots

    # Subplot 1: condition
    ax1 = plt.subplot(gs[2])

    plt.pcolormesh(raw_quest_cond_k[0,...,0], raw_quest_cond_k[0,...,1], raw_quest_cond_v[0,...,0], 
                   shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
    mask = np.abs(data_quest_cond_mask[0,:] - 1) < 0.01 # around 1
    print(data_quest_cond_k[0,mask,1].shape)
    plt.plot(data_quest_cond_k[0,mask,1], data_quest_cond_k[0,mask,2], 's', markersize=3, color = 'black', alpha = 0.7) 
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.title("question\ncondition")

    # Subplot 2: ground truth
    ax2 = plt.subplot(gs[3])
    plt.pcolormesh(raw_quest_qoi_k[0,...,0], raw_quest_qoi_k[0,...,1], raw_quest_qoi_v[0,...,0], 
                   shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
    plt.xlabel(r'$t$')
    plt.yticks([])
    plt.title("question\n QoI ground truth")

    # Subplot 3: prediction
    ax3 = plt.subplot(gs[4])
    plt.pcolormesh(raw_quest_qoi_k[0,...,0], raw_quest_qoi_k[0,...,1], pred[...,0], 
                   shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
    plt.xlabel(r'$t$')
    plt.yticks([])
    plt.title("question\n QoI prediction")

    # Subplot 4: prediction error
    ax4 = plt.subplot(gs[5])
    plt.pcolormesh(raw_quest_qoi_k[0,...,0], raw_quest_qoi_k[0,...,1], pred[...,0] - raw_quest_qoi_v[0,...,0], 
                   shading = shading, vmin = evmin, vmax = evmax, cmap = errcmap)
    plt.xlabel(r'$t$')
    plt.yticks([])
    plt.title("question\n QoI error")

    # Create shared color bar for subplots 1, 2, and 3
    cbar_ax1 = plt.subplot(gs[0])  # Position of the shared color bar
    cbar1 = fig.colorbar(ax1.collections[0], cax=cbar_ax1)
    cbar_ax1.yaxis.set_ticks_position('left') 

    # Create separate color bar for subplot 4
    cbar_ax2 = plt.subplot(gs[6])  # Separate color bar axes
    cbar2 = fig.colorbar(ax4.collections[0], cax=cbar_ax2)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(f"problem_{plot_index}_question.png", dpi = 600, bbox_inches='tight')
    

    # plot demos
    raw_demo_k = np.concatenate([raw_demo_cond_k, raw_demo_qoi_k], axis = 1) # [demo_num, 51, 100, 2]
    raw_demo_v = np.concatenate([raw_demo_cond_v, raw_demo_qoi_v], axis = 1) # [demo_num, 51, 100, 1]
    for demo_i in range(demo_num):
      plt.close('all')
      plt.figure(figsize=(3,3))
      
      plt.pcolormesh(raw_demo_k[demo_i,...,0], raw_demo_k[demo_i,...,1], raw_demo_v[demo_i,...,0], 
                     shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
      

      mask = np.abs(data_demo_cond_mask[demo_i,:] - 1) < 0.01 # around 1
      print(data_demo_cond_k[demo_i,mask,1].shape)
      plt.plot(data_demo_cond_k[demo_i,mask,1], data_demo_cond_k[demo_i,mask,2], 'o', markersize=3, color = 'blue', alpha = 0.7) 

      mask = np.abs(data_demo_qoi_mask[demo_i,:] - 1) < 0.01 # around 1
      print(data_demo_qoi_k[demo_i,mask,1].shape)
      plt.plot(data_demo_qoi_k[demo_i,mask,1], data_demo_qoi_k[demo_i,mask,2], 'o', markersize=3, color = 'red', alpha = 0.7) 

      
      plt.xlabel(r'$t$')
      plt.ylabel(r'$x$')
      plt.title(f"example #{demo_i + 1}")
      plt.tight_layout()
      plt.savefig(f"problem_{plot_index}_demo_{demo_i}.png", dpi = 600, bbox_inches='tight')


def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  tf.random.set_seed(FLAGS.seed + 123) 
  result_dict = run_analysis()
  for i in range(FLAGS.batch_size):
    plot_2D(result_dict, i)


if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_boolean('tfboard', False, 'dump into tfboard')

  flags.DEFINE_enum('task', 'ind', ['ind', 'ood', 'len', 'weno_quadratic', 'weno_cubic'], 'task type')
  flags.DEFINE_enum('backend', 'jax', ['jax','torch'], 'backend of runner')

  flags.DEFINE_integer('seed', 1, 'random seed')

  flags.DEFINE_integer('demo_num', 3, 'demo num, max 5')
  flags.DEFINE_float('vmin', 0.0, 'vmin')
  flags.DEFINE_float('vmax', 2.0, 'vmax')
  flags.DEFINE_float('evmin', -0.04, 'error vmin')
  flags.DEFINE_float('evmax', 0.04, 'error vmax')

  flags.DEFINE_list('test_data_dirs', '/home/shared/icon/data/data0910c', 'directories of testing data')
  flags.DEFINE_list('test_data_globs', ['train_mfc_gparam_hj_forward22*'], 'filename glob patterns of testing data')
  flags.DEFINE_string('test_config_filename', 'test_lm_plot_mfc_config.json', 'config file for testing')
  flags.DEFINE_list('test_demo_num_list', [1,2,3,4,5], 'demo number list for testing')
  flags.DEFINE_list('test_caption_id_list', [-1], 'caption id list for testing')

  flags.DEFINE_integer('batch_size', 8, 'batch size')
  flags.DEFINE_list('loss_mode', ['nocap'], 'loss mode')
  flags.DEFINE_list('write', [], 'write mode')

  flags.DEFINE_string('model', 'icon_lm', 'model name')
  flags.DEFINE_string('model_config_filename', '../config_model/model_lm_config.json', 'config file for model')
  flags.DEFINE_string('analysis_dir', '/home/shared/icon/analysis/icon_lm_learn_s1-20231005-094726', 'write file to dir')
  flags.DEFINE_string('results_name', '', 'additional file name for results')
  flags.DEFINE_string('restore_dir', '/home/shared/icon/save/user/ckpts/icon_lm_learn/20231005-094726', 'restore directory')
  flags.DEFINE_integer('restore_step', 1000000, 'restore step')


  app.run(main)
