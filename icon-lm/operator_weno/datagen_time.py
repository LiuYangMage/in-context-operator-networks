import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.append('../')
sys.path.append('../data_preparation/')
sys.path.append('../data_preparation/weno/')

import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt


import data_utils
from weno import weno_scheme, weno_roll



def generate_weno_scalar_sol(dx, dt, init, fn, steps, grad_fn = None, stable_tol = None):
  '''
  init: (batch, N, 1)
  '''
  alpha = weno_scheme.get_scalar_alpha_batch(init, grad_fn, 100, 0.1) # (batch,)
  left_bound = jnp.zeros_like(init) # dummy
  right_bound = jnp.zeros_like(init) # dummy
  us = [init]
  # warm up
  for _ in range(100):
    weno_scheme.weno_step_batch(dt, dx, us[-1], weno_scheme.get_w_classic, weno_roll.get_u_roll_periodic, fn, 
                          alpha, 'rk4', left_bound, right_bound)
  # timing and repeat
  utils.timer.tic("sim")
  for j in range(FLAGS.repeat):
    us = [init]
    for _ in range(steps):
      us.append(weno_scheme.weno_step_batch(dt, dx, us[-1], weno_scheme.get_w_classic, weno_roll.get_u_roll_periodic, fn, 
                            alpha, 'rk4', left_bound, right_bound))
  utils.timer.toc("sim")
  print("repeat = {}, steps = {}, time = {:.3f}".format(FLAGS.repeat, steps, utils.timer.get_time("sim")))
  print("average time for each simulation {:.3f}".format(utils.timer.get_time("sim") / FLAGS.repeat))
  out = jnp.stack(us, axis = 1) # (batch, steps + 1, N, 1)
  return out

def simulate_conservation_weno_cubic(seed, length, steps, dt, bs):
  '''
  simulate the conservation law with cubic flux function using WENO scheme
  du/dt + d(a * u^2 + b * u)/dx = 0
  a, b, c, specified in eqn_mode
  '''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeff_a = 1.0
  coeff_b = 1.0
  coeff_c = 1.0
  
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  init = data_utils.generate_gaussian_process(next(rng), xs, bs,  kernel = data_utils.rbf_circle_kernel_jax, 
                                                    k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N, 1)
  
  fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
  grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
  
  print("initial condition generated")
  sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn, stable_tol = 10.0) # (num, steps + 1, N, 1)
  return sol

def main(argv):
  seed = 1234
  eqns = 1
  length = 100
  steps = 200
  dt = 0.0005
  sim_num = 1 # simulation trajectories, i.e. number of initial conditions

  eqn_mode =  "fix_0.20_0.20_0.20", 
  sol = simulate_conservation_weno_cubic(seed, length, steps, dt, sim_num)
  
  print(sol.shape, sol.dtype)
  # (out_bs,) (out_bs, out_examples, 100, 1) 0.20000000_0.20000000_0.20000000


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_integer('repeat', 1000, 'repeat')
  app.run(main)


'''
CUDA_VISIBLE_DEVICES="0" python3 datagen_time.py 

initial condition generated
sim Took 16.3644 seconds
repeat = 1000, steps = 200, time = 16.364
average time for each simulation 0.016
(1, 201, 100, 1) float32
'''