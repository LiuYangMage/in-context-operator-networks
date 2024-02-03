
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from pprint import pprint

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from functools import partial
import haiku as hk
import optax
from absl import app, flags, logging
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape

import sys
sys.path.append('../')
import utils
import plot
import matplotlib.pyplot as plt

import plot_weno_new as new


label_dict_1 = {'sin_1.0_-1.0_1.0_1.0': "$\partial_t u + \partial_x(\sin(u) - \cos(u)) = 0$",
                'adaptive_sin_1.0_-1.0_1.0_1.0': "adaptive cubic fit",
              'cubic_-0.16666666667_0.5_1': "$\partial_t u + \partial_x(-1/6x^3 + 0.5x^2 + x) = 0$\nTaylor expansion",
              'cubic_-0.157_0.465_0.998': "$\partial_t u + \partial_x(-0.157x^3 + 0.465x^2 + 0.998x) = 0$\ncubic fit in [-1, 1]",
              'cubic_-0.132_0.370_0.971': "$\partial_t u + \partial_x(-0.132x^3 + 0.370x^2 + 0.971x) = 0$\ncubic fit in [-2, 2]"
              }
label_dict = {'sin_1.0_-1.0_1.0_1.0': "$\partial_t u + \partial_x(\sin(u) - \cos(u)) = 0$",
              'adaptive_sin_1.0_-1.0_1.0_1.0': "adaptive cubic fit",
              'cubic_-0.16666666667_0.5_1': "cubic Taylor polynomial",
              'cubic_-0.157_0.465_0.998': "cubic fit in [-1, 1]",
              'cubic_-0.132_0.370_0.971': "cubic fit in [-2, 2]",
              
              'tanh_1.0_1.0': "$\partial_t u + \partial_x(tanh(u)) = 0$",
              'adaptive_tanh_1.0_1.0': "adaptive cubic fit",
              'cubic_-0.33333_0.0_1.0': "cubic Taylor polynomial",
              'cubic_-0.225_0.0_0.979': "cubic fit in [-1, 1]",
              'cubic_-0.104_0.0_0.864': "cubic fit in [-2, 2]"
              }
color_dict = {'sin_1.0_-1.0_1.0_1.0': "k-",
              'adaptive_sin_1.0_-1.0_1.0_1.0': "b--",
              'cubic_-0.16666666667_0.5_1': "r--",
              'cubic_-0.157_0.465_0.998': "m--",
              'cubic_-0.132_0.370_0.971': "g--",

              'tanh_1.0_1.0': "k-",
              'adaptive_tanh_1.0_1.0': "b--",
              'cubic_-0.33333_0.0_1.0': "r--",
              'cubic_-0.225_0.0_0.979': "m--",
              'cubic_-0.104_0.0_0.864': "g--",
              }
figsize = (4,3.5)

def make_plot_forward(real_name, compare_list, ymax):

  model_config = utils.load_json("../config_model/model_lm_config.json")
  model_config['caption_len'] = 0

  restore_dir = '/home/shared/icon/save/user/ckpts/icon_weno/{}'.format(FLAGS.ckpt)
  restore_step = 1000000
  folder = f"weno_{real_name}_init_scale_{FLAGS.init_scale}_{FLAGS.groups}x{FLAGS.num}"
  if not os.path.exists(folder):
    os.makedirs(folder)
  data, runner = new.get_data_runner(model_config, restore_dir, restore_step, bs = FLAGS.num)
  rng = hk.PRNGSequence(jax.random.PRNGKey(1))

  stride = 5; mode = 'vanilla'
  sols = {}
  
  for name in compare_list:
    file_name = f'{folder}/sim_{name}.npy'
    key = (name,"sim")
    print(file_name, key)
    try:
      # direct simulation
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key], fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                                      num = FLAGS.num, groups = FLAGS.groups)
      sols[key] = sols[key][:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
      np.save(file_name, sols[key])

    file_name = f'{folder}/pred_forward_demo_{name}_init_{name}_stride{stride}_{mode}.npy'
    key = (name, name, stride, mode)
    print(file_name, key)
    try:
      # forward prediction with "name" record (t = 0 to 0.1) and "name" initial condition (at t = 0.1)
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key] = new.get_predict_forward(data, runner, sols[(name,"sim")], 
                                          FLAGS.ref, stride, mode, rng)
      np.save(file_name, sols[key])

    file_name = f'{folder}/pred_forward_demo_{name}_init_{real_name}_stride{stride}_{mode}.npy'
    key = (name, real_name, stride, mode)
    print(file_name, key)
    try:
      # forward prediction with "name" record (t = 0 to 0.1) and "real_name" initial condition (at t = 0.1)
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key] = new.get_predict_forward(data, runner, sols[(real_name,"sim")], 
                                     FLAGS.ref, stride, mode, rng, sols[(name,"sim")])
      np.save(file_name, sols[key])



  
  t = np.linspace(0.0, 0.5, 51, endpoint=True)
  plt.figure(figsize=figsize)
  for name in compare_list:
    linestyle = color_dict[name]
    plt.plot(t, new.get_error(sols[(name, "sim")], sols[(name, name, stride, mode)]), linestyle, label = "{}".format(label_dict[name]))
  
  plt.legend(loc = 'upper left')
  plt.ylim(0, ymax)
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.tight_layout()
  plt.savefig(f'{folder}/error_forward_{folder}_1.pdf')

  plt.figure(figsize=figsize)
  for name in compare_list:
    linestyle = color_dict[name]
    plt.plot(t, new.get_error(sols[(real_name,"sim")], sols[(name, real_name, stride, mode)]), linestyle, label = "{}".format(label_dict[name]))

  plt.legend(loc = 'lower right')
  plt.ylim(0, ymax)
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.tight_layout()
  plt.savefig(f'{folder}/error_forward_{folder}_2.pdf')


def make_plot_backward(real_name, compare_list, ymax):

  model_config = utils.load_json("../config_model/model_lm_config.json")
  model_config['caption_len'] = 0

  restore_dir = '/home/shared/icon/save/user/ckpts/icon_weno/{}'.format(FLAGS.ckpt)
  restore_step = 1000000

  folder = f"weno_{real_name}_init_scale_{FLAGS.init_scale}_{FLAGS.groups}x{FLAGS.num}"
  if not os.path.exists(folder):
    os.makedirs(folder)
  data, runner = new.get_data_runner(model_config, restore_dir, restore_step, bs = FLAGS.num)
  rng = hk.PRNGSequence(jax.random.PRNGKey(1))

  stride = 5; mode = 'vanilla'
  sols = {}
  
  for name in compare_list:

    file_name = f'{folder}/sim_{name}.npy'
    key = (name, "sim")
    print(file_name, key)
    try:
      #1 direct simulation
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key], fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                            num = FLAGS.num, groups = FLAGS.groups)
      sols[key] = sols[key][:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
      np.save(file_name, sols[key])


    file_name = f'{folder}/pred_backward_demo_{name}_init_{name}_stride{stride}_{mode}.npy'
    key = (name, name, stride, mode)
    print(file_name, key)
    try:
      #2 backward prediction with "name" record (t = 0.4 to 0.5) and "name" terminal condition (at t = 0.4)
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key] = new.get_predict_backward(data, runner, sols[(name,"sim")], 
                                           FLAGS.ref, stride, mode, rng)
      np.save(file_name, sols[key])


    file_name = f'{folder}/pred_backward_demo_{name}_init_{name}_sim_{name}_stride{stride}_{mode}.npy'
    key = (name, name, name, stride, mode)
    print(file_name, key)
    try:
      #3 backward prediction with "name" record (t = 0.4 to 0.5) and "name" terminal condition (at t = 0.4)
      # then apply "name" equation and simulate to t = 0.4
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      # set steps = 1, not 0, due to jax issue
      _, fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = 1, dt = FLAGS.dt, 
                                            num = FLAGS.num, groups = FLAGS.groups)
      sols[key] = new.get_backward_consistency(sols[(name, name, stride, mode)], FLAGS.ref, fn, grad_fn)
      np.save(file_name, sols[key])


    file_name = f'{folder}/pred_backward_demo_{name}_init_{real_name}_stride{stride}_{mode}.npy'
    key = (name, real_name, stride, mode)
    print(file_name, key)
    try:
      #4 backward prediction with "name" record (t = 0.4 to 0.5) and "real_name" terminal condition (at t = 0.4)
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      sols[key] = new.get_predict_backward(data, runner, sols[(real_name,"sim")], FLAGS.ref, 
                                           stride, mode, rng, sols[(name,"sim")])
      np.save(file_name, sols[key])


    file_name = f'{folder}/pred_backward_demo_{name}_init_{real_name}_sim_{real_name}_stride{stride}_{mode}.npy'
    key = (name, real_name, real_name, stride, mode)
    print(file_name, key)
    try:
      #5 backward prediction with "name" record (t = 0.4 to 0.5) and "real_name" terminal condition (at t = 0.4)
      # then apply "real_name" equation and simulate to t = 0.4
      assert not FLAGS.regen
      sols[key] = np.load(file_name)
    except:
      _, fn, grad_fn = new.get_data(real_name, FLAGS.length, 1, FLAGS.dt, FLAGS.num, FLAGS.groups)
      sols[key] = new.get_backward_consistency(sols[(name, real_name, stride, mode)], FLAGS.ref, fn, grad_fn)
      np.save(file_name, sols[key])

  t = np.linspace(0.5, 0, 51, endpoint=True)
  plt.figure(figsize=figsize)
  for name in compare_list:
    linestyle = color_dict[name]
    error = new.get_error(sols[(name, name, name, stride, mode)], 
                          sols[(name, 'sim')][:,:,-(FLAGS.ref+1):-FLAGS.ref,:,:])
    error[:FLAGS.ref] = 0
    plt.plot(t, error, linestyle, label = "{}".format(label_dict[name]))
  
  plt.legend(loc = 'upper right')
  plt.ylim(0, ymax)
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.tight_layout()
  plt.savefig(f'{folder}/error_backward_{folder}_1.pdf')

  t = np.linspace(0.5, 0, 51, endpoint=True)
  plt.figure(figsize=figsize)
  for name in compare_list:
    linestyle = color_dict[name]
    error = new.get_error(sols[(name, real_name, real_name, stride, mode)], 
                          sols[(real_name, 'sim')][:,:,-(FLAGS.ref+1):-FLAGS.ref,:,:])
    error[:FLAGS.ref] = 0
    plt.plot(t, error, linestyle, label = "{}".format(label_dict[name]))
  
  plt.legend(loc = 'lower left')
  plt.ylim(0, ymax)
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.tight_layout()
  plt.savefig(f'{folder}/error_backward_{folder}_2.pdf')


def main(argv):
  del argv

  real_name_sin1 = 'sin_1.0_-1.0_1.0_1.0'

  compare_list_sin1 = ['sin_1.0_-1.0_1.0_1.0',
                'adaptive_sin_1.0_-1.0_1.0_1.0',
                'cubic_-0.16666666667_0.5_1',
                'cubic_-0.157_0.465_0.998',
                'cubic_-0.132_0.370_0.971'
                ]
  
  real_name_tanh1 = 'tanh_1.0_1.0'

  compare_list_tanh1 = ['tanh_1.0_1.0',
                'adaptive_tanh_1.0_1.0',
                'cubic_-0.33333_0.0_1.0',
                'cubic_-0.225_0.0_0.979',
                'cubic_-0.104_0.0_0.864'
                ]
  

  if "sin1_f" in FLAGS.mode:
    make_plot_forward(real_name_sin1, compare_list_sin1, ymax = 0.04)
  if "sin1_b" in FLAGS.mode:
    make_plot_backward(real_name_sin1, compare_list_sin1, ymax = 0.04)

  if "tanh1_f" in FLAGS.mode:
    make_plot_forward(real_name_tanh1, compare_list_tanh1, ymax = 0.08)
  if "tanh1_b" in FLAGS.mode:
    make_plot_backward(real_name_tanh1, compare_list_tanh1, ymax = 0.08)
  

if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_list('mode', ["sin1_f", "sin1_b", "tanh1_f", "tanh1_b"], 'run mode')
  
  flags.DEFINE_integer('seed', 200, 'random seed')
  flags.DEFINE_integer('steps', 1000, 'training steps')
  flags.DEFINE_float('dt', 0.0005, 'dt')
  flags.DEFINE_float('init_scale', -1.0, 'the scale of initial condition')
  flags.DEFINE_integer('downsample', 20, 'downsample to build sequence')
  flags.DEFINE_integer('length', 100, 'length of grid')
  flags.DEFINE_integer('ref', 10, 'ref steps, based on downsampled sequence')
  flags.DEFINE_integer('stride', 5, 'prediction stride, based on downsampled sequence')
  flags.DEFINE_string('demo_mode', "random", 'mode for demo example selection')
  flags.DEFINE_integer('groups', 8, 'groups')
  flags.DEFINE_integer('num', 64, 'num')
  flags.DEFINE_bool('regen', False, 'regenerate data')
  flags.DEFINE_string('ckpt', "20231209-222440", 'checkpoint')


  app.run(main)
