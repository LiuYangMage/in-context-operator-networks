
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
from scipy.optimize import curve_fit
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

FLAGS = flags.FLAGS

utils.set_seed(1234)
devices = len(jax.devices())


def get_data_runner(model_config, restore_dir, restore_step, bs):
  length = FLAGS.length
  optimizer = optax.adamw(0.0001) # dummy optimizer
  # template of data, with shared k and mask
  k = np.linspace(0.0, 1.0, length, endpoint=False)[None,:,None]
  k = np.pad(k, ((0,0),(0,0),(2,0)), mode='constant', constant_values=0.0) # (1, N, 3)
  demo_cond_k = np.tile(k, [5,1,1]) # (5, N, 3)
  demo_qoi_k = np.tile(k, [5,1,1]) # (5, N, 3)
  quest_cond_k = np.tile(k, [1,1,1]) # (1, N, 3)
  quest_qoi_k = np.tile(k, [1,1,1]) # (1, N, 3)
  demo_cond_mask = np.ones((5, length), dtype=bool)
  demo_qoi_mask = np.ones((5, length), dtype=bool)
  quest_cond_mask = np.ones((1, length), dtype=bool)
  quest_qoi_mask = np.ones((1, length), dtype=bool)

  data = Data(input_id = np.zeros((0,)), 
              embedding_raw = np.zeros((0,)), 
              embedding_pool = np.zeros((0,)), 
              embedding_mask = np.zeros((0,)), 
              demo_cond_k = demo_cond_k,
              demo_cond_v = np.zeros((5,length,1)),
              demo_cond_mask = demo_cond_mask,
              demo_qoi_k = demo_qoi_k,
              demo_qoi_v = np.zeros((5,length,1)),
              demo_qoi_mask = demo_qoi_mask,
              quest_cond_k = quest_cond_k,
              quest_cond_v = np.zeros((1,length,1)),
              quest_cond_mask = quest_cond_mask,
              quest_qoi_k = quest_qoi_k,
              quest_qoi_mask = quest_qoi_mask,
              )

  add_dim = lambda x: np.reshape(np.repeat(x[None,...], bs, axis=0), (devices, bs//devices, *x.shape))
  data = tree.tree_map(add_dim, data)
  print(tree.tree_map(lambda x: x.shape, data)) 

  runner = Runner_lm(seed = 0,
                  model = "icon_lm",
                  data = data,
                  model_config = model_config,
                  optimizer = optimizer,
                  trainable_mode = 'all',
                  loss_mode = ["nocap"],
                  print_model = False
                  )

  runner.restore(restore_dir, restore_step, restore_opt_state=False)
  return data, runner

def generate_conservation_weno(seed, length, steps, dt, num, fn, grad_fn, init = None):

  rng = hk.PRNGSequence(jax.random.PRNGKey(seed + FLAGS.seed))
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)

  if init is None:
    while True:
      init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None]
      if jnp.max(jnp.abs(init)) < 3.0:
        break
    if FLAGS.init_scale > 0:
      init_min, init_max = jnp.min(init, axis = (-1,-2), keepdims = True), jnp.max(init, axis = (-1,-2), keepdims = True)
      init = (init - init_min) / (init_max - init_min) * 2.0 - 1.0 # normalize to [-1, 1]
      init = init * FLAGS.init_scale # scale
  # init: (num, N, 1), N = 101, sol: (num, steps + 1, N, 1)
  sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn, stable_tol = 10.0)
  return sol

def cubic_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def cubic_function_grad(x, a, b, c, d):
    return 3*a*x**2 + 2*b*x + c

def get_fn(name_split):
  if name_split[0] == 'sin':
    a, b, c, d = [float(x) for x in name_split[1:]]
    print("sin", a, b, c, d)
    fn = jax.jit(lambda u: a * jnp.sin(c * u) + b * jnp.cos(d * u))
    grad_fn = jax.jit(lambda u: a * c * jnp.cos(c * u) - b * d * jnp.sin(d * u))
  elif name_split[0] == 'cubic':
    a = float(name_split[1])
    b = float(name_split[2])
    c = float(name_split[3])
    print("cubic", a, b, c)
    fn = jax.jit(lambda u: a * u * u * u + b * u * u + c * u)
    grad_fn = jax.jit(lambda u: 3 * a * u * u + 2 * b * u + c)
  elif name_split[0] == 'tanh':
    a, b = [float(x) for x in name_split[1:]]
    print("tanh", a, b)
    fn = jax.jit(lambda u: a * jnp.tanh(b*u))
    grad_fn = jax.jit(lambda u: a * b * (1 - jnp.tanh(b*u) ** 2))
  else:
    raise NotImplementedError
  return fn, grad_fn

def get_data(name, length, steps, dt, num, groups, init = None):
  if init is None:
    init = [None for _ in range(groups)]

  if 'adaptive' in name: # e.g. 'adaptive_sin_1.0_-1.0_1.0_1.0'
    name_split = name.split('_')[1:]
    print("adaptive", name_split)
  else:
    name_split = name.split('_')

  fn, grad_fn = get_fn(name_split)

  sols = [None for _ in range(groups)]
  for seed in range(groups):
    sols[seed] = generate_conservation_weno(seed = seed, length = length, steps = steps, dt = dt, num = num, 
                                            fn = fn, grad_fn = grad_fn, init = init[seed]) # (num, steps + 1, N, 1)
  sols = np.stack(sols, axis = 0) # (groups, num, steps + 1, N, 1)

  if 'adaptive' in name:
    new_sols = [[None for j in range(sols.shape[1])] for i in range(sols.shape[0])]
    new_fns = [[None for j in range(sols.shape[1])] for i in range(sols.shape[0])]
    new_grad_fns = [[None for j in range(sols.shape[1])] for i in range(sols.shape[0])]
    for i in range(sols.shape[0]):
      for j in range(0, sols.shape[1]):
        this_init = sols[i,j,0,:,:] # (N, 1)
        this_min, this_max = np.min(this_init), np.max(this_init)
        this_x = np.linspace(this_min, this_max, 100)
        this_y = fn(this_x)
        params, _ = curve_fit(cubic_function, this_x, this_y) # best cubic fit in this range
        new_fn = jax.jit(lambda u: cubic_function(u, *params))
        new_grad_fn = jax.jit(lambda u: cubic_function_grad(u, *params))
        new_sols[i][j] = generate_conservation_weno(seed = i, length = length, steps = steps, dt = dt, num = None, 
                                                    fn = new_fn, grad_fn = new_grad_fn, init = this_init[None,...])[0] # (steps + 1, N, 1)
        new_fns[i][j] = new_fn
        new_grad_fns[i][j] = new_grad_fn
        print(i, j, f"range [{this_min}, {this_max}]", params, flush=True)
    new_sols = np.array(new_sols) # (groups, num, steps + 1, N, 1)
    sols = new_sols
    fn = new_fns
    grad_fn = new_grad_fns
        
  return sols, fn, grad_fn


# %%
@partial(jax.jit, static_argnums=(2,3,))
def get_demo(key, sol, stride, mode = 'random'):
  # sol: (steps, 100, 1)
  demo_cond_v = sol[:-stride,:,:] # (s,N,1)
  demo_qoi_v = sol[stride:,:,:] # (s,N,1)
  if mode == 'random':
    idx = jax.random.choice(key, np.arange(demo_cond_v.shape[0]), shape=(5,), replace=False) # (5,)
  elif mode == 'order':
    idx = jnp.arange(5, dtype=int)
  else:
    raise ValueError("mode {} not supported".format(mode))
  demo_cond_v = demo_cond_v[idx, :, :] # (5, N, 1)
  demo_qoi_v = demo_qoi_v[idx, :, :] # (5, N, 1)
  return demo_cond_v, demo_qoi_v, idx

get_demo_batch = jax.jit(jax.vmap(get_demo, in_axes=(0, 0, None, None), out_axes=(0, 0, 0)), static_argnums=(2,3,))


def get_predict(data, runner, demo_cond_v, demo_qoi_v, quest_cond_v, mode = 'vanilla'):
  # demo_cond_v, demo_qoi_v: (bs, 5, N, 1)
  # quest_cond_v: (bs, 1, N, 1)
  split_device = lambda x: np.reshape(x, (devices, x.shape[0]//devices, *x.shape[1:]))
  demo_cond_v = split_device(demo_cond_v)
  demo_qoi_v = split_device(demo_qoi_v)
  quest_cond_v = split_device(quest_cond_v)
  if mode == 'vanilla':
    data = data._replace(demo_cond_v = demo_cond_v, demo_qoi_v = demo_qoi_v, quest_cond_v = quest_cond_v)
    pred = runner.get_pred(data, with_caption = False)[:,:,None,:,:]
  elif 'minmax' in mode:
    scale = float(mode.split('_')[-1])
    this_data = np.concatenate([demo_cond_v, demo_qoi_v, quest_cond_v], axis = -3) # (ps, bs, 11, N, 1)
    data_max = np.max(this_data, axis = (-1,-2,-3), keepdims = True) # (ps, bs, 1, 1, 1)
    data_min = np.min(this_data, axis = (-1,-2,-3), keepdims = True) # (ps, bs, 1, 1, 1)
    mean = (data_max + data_min) / 2
    std = (data_max - data_min) / 2
    std = std / scale # so that demo and question are in [-scale, scale]
    demo_cond_v = (demo_cond_v - mean) / std
    demo_qoi_v = (demo_qoi_v - mean) / std
    quest_cond_v = (quest_cond_v - mean) / std
    data = data._replace(demo_cond_v = demo_cond_v, demo_qoi_v = demo_qoi_v, quest_cond_v = quest_cond_v)
    pred = runner.get_pred(data, with_caption = False)[:,:,None,:,:]
    pred = pred * std + mean
  else:
    raise ValueError("mode {} not supported".format(mode))
  pred = einshape('pb...->(pb)...', pred) # (bs, 1, N, 1)
  return pred



def get_predict_forward(data, runner, sols, ref, stride, mode, rng, sols_demo = None):
  '''
  recursive forward prediction
  data: template of data, with shared k and mask
  runner: model runner
  sols: (groups, bs, 51, 100, 1), Delta t = 0.01
  ts: [0,0.01,0.02,...,0.5]
  ref: 10, data
  stride: 5, prediction stride
  mode for predicition, e.g. vanilla, normalize_1.0, minmax_1.0
  rng: random key
  sols_demo: used for demo construction, if None, use sols
  '''
  if sols_demo is None:
    sols_demo = sols
  preds = sols.copy() # (groups, bs, 51, 100, 1)
  preds[:,:,ref+1:,:,:] = 0.0
  for gid in range(sols.shape[0]):
    demo_record = sols_demo[gid,:,0:ref+1,:,:] # [0,0.01,0.02,...,0.1], (bs, 11, 100, 1)
    print(f"group {gid}:", end = ' ', flush = True)
    for st in range(1, stride):
      key = jax.random.split(next(rng), FLAGS.num)
      demo_cond_v, demo_qoi_v, idx = get_demo_batch(key, demo_record, st, FLAGS.demo_mode) # (bs, 5, 100, 1), (bs, 5, 100, 1), (bs, 5)
      quest_cond_v = sols[gid,:,ref:ref+1,:,:] # (bs, 1, 100, 1)
      this_pred = get_predict(data, runner, demo_cond_v, demo_qoi_v, quest_cond_v, mode)
      preds[gid,:,ref+st,:,:] = this_pred[:,0,:,:]
      print(ref+st, end = ' ', flush = True)
    # fix stride
    key = jax.random.split(next(rng), FLAGS.num)
    demo_cond_v, demo_qoi_v, idx = get_demo_batch(key, demo_record, stride, FLAGS.demo_mode)
    for idx in range(ref+stride, sols.shape[2]):
      quest_cond_v = preds[gid,:,idx-stride:idx-stride+1,:,:]
      this_pred = get_predict(data, runner, demo_cond_v, demo_qoi_v, quest_cond_v, mode)
      preds[gid,:,idx,:,:] = this_pred[:,0,:,:]
      print(idx, end = ' ', flush = True)
    print("")
  return preds


def get_predict_backward(data, runner, sols, ref, stride, mode, rng, sols_demo = None):
  '''
  flip sols, then use get_predict_forward
  preds: (groups, bs, 51, 100, 1), [0.5,0.49,...,0.0]
  '''
  if sols_demo is None:
    sols_demo = sols
  flip_sols = np.zeros_like(sols)
  flip_sols[:,:,0:ref+1,:,:] = sols[:,:,-1:-(ref+2):-1,:,:] # [0.5,0.49,...,0.4], (groups, bs, 11, 100, 1)
  flip_sols_demo = np.zeros_like(sols_demo)
  flip_sols_demo[:,:,0:ref+1,:,:] = sols_demo[:,:,-1:-(ref+2):-1,:,:] # [0.5,0.49,...,0.4], (groups, bs, 11, 100, 1)
  preds = get_predict_forward(data, runner, flip_sols, ref, stride, mode, rng, flip_sols_demo)
  return preds


def get_error(sol1, sol2):
  return np.mean(np.abs(sol1 - sol2), axis = (0,1,3,4))


def get_backward_consistency(sols, ref, fn, grad_fn):
  '''
  sol: (groups, bs, 51, 100, 1), [0.5,0.49,...,0.0]
  ref: 10, i.e., 0-10 as reference
  apply equation to sols at each time step, and simulate to t = 0.4, store in consist
  '''
  length = FLAGS.length
  dt = FLAGS.dt
  downsample = FLAGS.downsample
  consist = np.zeros_like(sols) # represent u(0.4) if we simulate from u(t)
  for gid in range(sols.shape[0]):
    print(f"group {gid}:", end = ' ', flush = True)
    for tid in range(ref, sols.shape[2]):
      init = sols[gid,:,tid,:,:] # (bs, 100, 1)
      steps = (tid - ref) * downsample
      if type(fn) is list:
        for bid in range(sols.shape[1]):
          sim = generate_conservation_weno(seed = 0, length = length, steps = steps, dt = dt, num = None, 
                                          fn = fn[gid][bid], grad_fn = grad_fn[gid][bid], init = init[bid:bid+1,:,:])
          consist[gid,bid,tid,:,:] = sim[0,-1,:,:]
      else:
        sim = generate_conservation_weno(seed = 0, length = length, steps = steps, dt = dt, num = None, 
                                      fn = fn, grad_fn = grad_fn, init = init) # (bs, steps + 1, N, 1)
        consist[gid,:,tid,:,:] = sim[:,-1,:,:] 
      print(tid, end = ' ', flush = True)
    print("")
  return consist


def main(argv):
  del argv

  
if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_list('mode', ["get","plot"], 'run mode')

  flags.DEFINE_integer('steps', 1000, 'training steps')
  flags.DEFINE_float('dt', 0.0005, 'dt')
  flags.DEFINE_float('init_scale', -1, 'the scale of initial condition')
  flags.DEFINE_integer('downsample', 20, 'downsample to build sequence')
  flags.DEFINE_integer('ref', 10, 'ref steps, based on downsampled sequence')
  flags.DEFINE_integer('stride', 5, 'prediction stride, based on downsampled sequence')
  flags.DEFINE_string('demo_mode', "random", 'mode for demo example selection')
  flags.DEFINE_integer('groups', 8, 'groups')
  flags.DEFINE_integer('num', 64, 'num')
  flags.DEFINE_bool('regen', False, 'regenerate data')
  flags.DEFINE_string('ckpt', "20231209-222440", 'checkpoint')

  app.run(main)


# %%
