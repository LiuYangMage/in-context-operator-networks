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


def generate_weno_scalar_sol(dx, dt, init, fn, steps, grad_fn, stable_tol, periodic):
  '''
  init: (batch, N, 1)
  '''
  alpha = weno_scheme.get_scalar_alpha_batch(init, grad_fn, 100, 0.1) # (batch,)
  if periodic:
    left_bound = jnp.zeros_like(init) # dummy
    right_bound = jnp.zeros_like(init) # dummy
    roll = weno_roll.get_u_roll_periodic
  else:
    print(init.shape, flush=True)
    left_bound = init[:,0,:]
    right_bound = init[:,-1,:]
    roll = weno_roll.get_u_roll_dirichlet
  us = [init]
  for i in range(steps):
    us.append(weno_scheme.weno_step_batch(dt, dx, us[-1], weno_scheme.get_w_classic, roll, fn, 
                          alpha, 'rk4', left_bound, right_bound))
  out = jnp.stack(us, axis = 1) # (batch, steps + 1, N, 1)
  # check if the solution is stable
  if stable_tol and (jnp.any(jnp.isnan(us[-1])) or jnp.max(jnp.abs(us[-1])) > stable_tol):
    raise ValueError("sol contains nan")
  return out


def simulate_conservation_weno_cubic(seed, eqns, length, steps, dt, num, eqn_mode):
  '''
  simulate the conservation law with cubic flux function using WENO scheme
  du/dt + d(a * u^2 + b * u)/dx = 0
  a, b, c, specified in eqn_mode
  '''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if 'random' in eqn_mode:
    minval = float(eqn_mode.split('_')[1])
    maxval = float(eqn_mode.split('_')[2])
    coeffs_a = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_b = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_c = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
  elif 'grid' in eqn_mode:
    minval = float(eqn_mode.split('_')[1])
    maxval = float(eqn_mode.split('_')[2])
    values = np.linspace(minval, maxval, eqns)
    coeffs_a, coeffs_b, coeffs_c = np.meshgrid(values, values, values)
    coeffs_a = coeffs_a.flatten()
    coeffs_b = coeffs_b.flatten()
    coeffs_c = coeffs_c.flatten()
  elif 'fix' in eqn_mode:
    val_0 = float(eqn_mode.split('_')[1])
    val_1 = float(eqn_mode.split('_')[2])
    val_2 = float(eqn_mode.split('_')[3])
    coeffs_a = np.array([val_0] * eqns)
    coeffs_b = np.array([val_1] * eqns)
    coeffs_c = np.array([val_2] * eqns)
  else:
    raise NotImplementedError("eqn_mode = {} is not implemented".format(eqn_mode))
  
  if 'dirichlet' in eqn_mode:
    periodic = False
  else:
    periodic = True
  print('periodic = {}'.format(periodic), flush=True)

  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    print("coeff_a = {:.3f}, coeff_b = {:.3f}, coeff_c = {:.3f}".format(coeff_a, coeff_b, coeff_c), flush=True)
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  all_xs = []; all_us = []; all_params = []
  
  # general 
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
    grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
    init = []
    for loops in range(num//100): # generate initial conditions in groups.
      while True:
        this_init = data_utils.generate_gaussian_process(next(rng), xs, 100,  kernel = data_utils.rbf_circle_kernel_jax, 
                                                    k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N, 1)
        if jnp.max(jnp.abs(this_init)) < 3.0:
          break
      init.append(this_init)
    init = jnp.concatenate(init, axis = 0) # (num, N, 1)
    print("initial condition generated")
    sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn, 
                                   stable_tol = 10.0, periodic = periodic) # (num, steps + 1, N, 1)
    all_xs.append(xs) # (N,)
    all_us.append(sol) # (num, steps + 1, N, 1)
    all_params.append("{:.8f}_{:.8f}_{:.8f}".format(coeff_a, coeff_b, coeff_c))
    utils.print_dot(i)

  return all_xs, all_us, all_params

def seperate_data(all_xs, all_us, seed, stride, out_bs, out_examples):
  all_u1s = []; all_u2s = []; all_x1s = []; all_x2s = []
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  for xs, us in zip(all_xs, all_us):
    u1 = einshape("ijkl->(ij)kl", us[:,:-stride,:,:]) # (sim_num * step, N, 1)
    u2 = einshape("ijkl->(ij)kl", us[:,stride:,:,:]) # (sim_num * step, N, 1)
    # shuffle the first dimension of u1 and u2, keep the same permutation between u1 and u2
    key = next(rng)
    u1 = jax.random.permutation(key, u1, axis = 0) # (num * step, N, 1)
    u2 = jax.random.permutation(key, u2, axis = 0) # (num * step, N, 1)
    
    # truncate and reshape u1 and u2 to (out_bs, out_examples, N, 1)
    u1 = u1[:out_bs*out_examples,:,:] # (out_bs * out_examples, N, 1)
    u2 = u2[:out_bs*out_examples,:,:] # (out_bs * out_examples, N, 1)
    u1 = einshape("(ij)kl->ijkl", u1, i = out_bs, j = out_examples) # (num, examples, N, 1)
    u2 = einshape("(ij)kl->ijkl", u2, i = out_bs, j = out_examples) # (num, examples, N, 1)

    x1 = einshape("k->ijkl", xs, i = out_bs, j = out_examples, l = 1) # (num, examples, N, 1)
    x2 = einshape("k->ijkl", xs, i = out_bs, j = out_examples, l = 1) # (num, examples, N, 1)

    all_u1s.append(u1); all_u2s.append(u2); all_x1s.append(x1); all_x2s.append(x2)

  return all_u1s, all_u2s, all_x1s, all_x2s


def main(argv):
  seed = 1234
  eqns = 1
  length = 100
  steps = 1000
  dt = 0.0005
  stride = 200
  sim_num = 1000 # simulation trajectories, i.e. number of initial conditions

  out_bs = 100 # save data batch size
  out_examples = 1001 # examples in each batch
  # the data shape would be (out_bs, out_examples, 100, 1), 
  # the last example would be used for testing, 
  # the first out_examples - 1 examples would be used for training/in-context examples
  
  eqn_mode_list = [
                   "fix_0.20_0.20_0.20", 
                   "fix_0.21_0.21_0.21",
                   "fix_0.25_0.25_0.25",
                   "fix_0.30_0.30_0.30",
                   "fixdirichlet_0.20_0.20_0.20", # dirichlet boundary condition
                   ]
  
  for eqn_mode in eqn_mode_list:
    all_xs, all_us, all_params = simulate_conservation_weno_cubic(seed, eqns, length, steps, dt, sim_num, eqn_mode)
    print(all_xs[0].shape, all_us[0].shape, all_params[0])
    # (out_bs,) (out_bs, out_examples, 100, 1) 0.20000000_0.20000000_0.20000000

    # save raw data
    with open("{}_raw.pkl".format(eqn_mode), "wb") as file:
      pickle.dump((all_xs, all_us, all_params), file)

    all_u1s, all_u2s, all_x1s, all_x2s = seperate_data(all_xs, all_us, seed+1, stride, out_bs, out_examples)
    print(all_u1s[0].shape, all_u2s[0].shape, all_x1s[0].shape, all_x2s[0].shape)
    # (out_bs, out_examples, 100, 1) (out_bs, out_examples, 100, 1) (out_bs, out_examples, 100, 1) (out_bs, out_examples, 100, 1)

    # save useful data
    with open("{}.pkl".format(eqn_mode), "wb") as file:
      pickle.dump((all_u1s, all_u2s, all_x1s, all_x2s, all_params), file)

if __name__ == '__main__':
  FLAGS = flags.FLAGS

  app.run(main)

