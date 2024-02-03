
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
from dataloader import Data, DataProvider, print_eqn_caption, split_data
import matplotlib.pyplot as plt

sys.path.append('../data_preparation')
import data_utils
sys.path.append('../data_preparation/weno/')
from weno_solver import generate_weno_scalar_sol

import pickle 
from runner_jax import Runner_lm
import plot_weno_new as new



def main(argv):
  del argv
  Delta_t = FLAGS.dt * FLAGS.downsample # Delta_t = 0.01

  model_config = utils.load_json("../config_model/model_lm_config.json")
  model_config['caption_len'] = 0

  restore_dir = '/home/shared/icon/save/user/ckpts/icon_weno/{}'.format(FLAGS.ckpt)
  restore_step = 1000000
  real_name = FLAGS.real_name
  name = real_name

  folder = f"weno_{real_name}_init_scale_{FLAGS.init_scale}_{FLAGS.groups}x{FLAGS.num}"
  if not os.path.exists(folder):
    os.makedirs(folder)

  data, runner = new.get_data_runner(model_config, restore_dir, restore_step, bs = FLAGS.num)
  rng = hk.PRNGSequence(jax.random.PRNGKey(1))

  stride = FLAGS.stride; mode = FLAGS.change_mode
  forward_sols = {}
  backward_sols = {}

  sols = forward_sols

  file_name = f'{folder}/sim_{name}.npy'
  key = (name,"sim")
  print(file_name, key)
  try:
    sols[key] = np.load(file_name)
  except:
    sol, fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                                    num = FLAGS.num, groups = FLAGS.groups)
    sols[key] = sol[:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
    np.save(file_name, sols[key])

  file_name = f'{folder}/pred_forward_demo_{name}_init_{name}_stride{stride}_{mode}.npy'
  key = (name, stride, mode)
  print(file_name, key)
  try:
    # forward prediction with "name" record (t = 0 to 0.1) and "name" initial condition (at t = 0.1)
    assert not FLAGS.regen
    sols[key] = np.load(file_name)
  except:
    sols[key] = new.get_predict_forward(data, runner, sols[(name,"sim")], 
                                        FLAGS.ref, stride, mode, rng)
    np.save(file_name, sols[key])

  
  sols = backward_sols
    
  file_name = f'{folder}/sim_{name}.npy'
  key = (name, "sim")
  try:
    # direct simulation
    assert not FLAGS.regen
    sols[key] = np.load(file_name)
  except:
    sol, fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                                    num = FLAGS.num, groups = FLAGS.groups)
    sols[key] = sol[:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
    np.save(file_name, sols[key])

  file_name = f'{folder}/pred_backward_demo_{name}_init_{name}_stride{stride}_{mode}.npy'
  key = (name, stride, mode)
  try:
    # backward prediction with "name" record (t = 0.4 to 0.5) and "name" terminal condition (at t = 0.4)
    assert not FLAGS.regen
    sols[key] = np.load(file_name)
  except:
    sols[key] = new.get_predict_backward(data, runner, sols[(name,"sim")], FLAGS.ref, stride, mode, rng)
    np.save(file_name, sols[key])


  file_name = f'{folder}/pred_backward_demo_{name}_init_{name}_simfromend_{name}_stride{stride}_{mode}.npy'
  key = (name, stride, mode, name)
  try:
    # backward prediction with "name" record (t = 0.4 to 0.5) and "name" terminal condition (at t = 0.5)
    # then simulate from from the end, i.e. t = 0.
    assert not FLAGS.regen
    sols[key] = np.load(file_name)
  except:
    sol, fn, grad_fn = new.get_data(name, length = FLAGS.length, steps = FLAGS.steps, dt = FLAGS.dt, 
                                  num = FLAGS.num, groups = FLAGS.groups, init = sols[(name, stride, mode)][:,:,-1,:,:])
    sols[key] = sol[:,:,::FLAGS.downsample,:,:] # (groups, bs, 51, 100, 1)
    np.save(file_name, sols[key])


  sols = forward_sols
  t = np.linspace(0.0, 0.5, 51, endpoint=True)
  x = np.linspace(0,1,100,endpoint=False)
  for gid, bid in zip([0]*10, list(range(10))):
    plt.figure(figsize=(3*6,3*1))
    for i in range(6):
      plt.subplot(1,6,i+1)
      idx = i * 10
      plt.plot(x, sols[(real_name, 'sim')][gid,bid,idx,:,0], 'k-', label = 'ground truth')
      if i > 1:
        plt.plot(x, sols[(real_name, stride, mode)][gid,bid,idx,:,0], 'r--', label = 'prediction')

      plt.xlabel('$x$')
      plt.ylabel('$u$')
      if i == 2:
        plt.legend()
      plt.title(f'$t$ = {t[idx]:.2f}')
    # plt.suptitle(label_dict[real_name])
    plt.tight_layout()
    plt.savefig(f'{folder}/profile_forward_{folder}_stride{stride}_{mode}_gid{gid}_bid{bid}.pdf')

  sols = backward_sols
  t = np.linspace(0.5,0.0, 51, endpoint=True)
  x = np.linspace(0,1,100,endpoint=False)
  for gid, bid in zip([0]*10, list(range(10))):
    plt.figure(figsize=(3*6,3*1))
    for i in range(6):
      plt.subplot(1,6,i+1)
      idx = i * 10
      if i <=1:
        plt.plot(x, sols[(real_name, stride, mode)][gid,bid,idx,:,0], 'k-', label = 'ground truth')
      if i > 1:
        plt.plot(x, sols[(real_name, stride, mode)][gid,bid,idx,:,0], 'r-', label = 'prediction')
      
      plt.plot(x, sols[(real_name, stride, mode, real_name)][gid,bid,50-idx,:,0], 'b--', label = 'forward simulation')
      plt.xlabel('$x$')
      plt.ylabel('$u$')
      if i == 2 or i == 1:
        plt.legend()
      plt.title(f'$t$ = {t[idx]:.2f}')
    # plt.suptitle(label_dict[real_name])
    plt.tight_layout()
    plt.savefig(f'{folder}/profile_backward_{folder}_stride{stride}_{mode}_gid{gid}_bid{bid}.pdf')


if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_string('real_name', 'sin_1.0_-1.0_1.0_1.0', 'name of function')

  flags.DEFINE_integer('seed', 200, 'random seed')
  flags.DEFINE_integer('steps', 1000, 'training steps')
  flags.DEFINE_float('dt', 0.0005, 'dt')
  flags.DEFINE_float('init_scale', -1.0, 'the scale of initial condition')
  flags.DEFINE_integer('downsample', 20, 'downsample to build sequence')
  flags.DEFINE_integer('length', 100, 'length of grid')
  flags.DEFINE_integer('ref', 10, 'ref steps, based on downsampled sequence')
  flags.DEFINE_integer('stride', 5, 'prediction stride, based on downsampled sequence')
  flags.DEFINE_string('change_mode', 'vanilla', 'change of variable mode')
  flags.DEFINE_string('demo_mode', "random", 'mode for demo example selection')
  flags.DEFINE_integer('groups', 1, 'groups')
  flags.DEFINE_integer('num', 64, 'num')
  flags.DEFINE_bool('regen', False, 'regenerate data')
  flags.DEFINE_string('ckpt', "20231209-222440", 'checkpoint')


  app.run(main)
