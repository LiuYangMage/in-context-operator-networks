import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

import data_dynamics as dyn
import data_series as series
import data_pdes as pdes
import data_mfc_hj as mfc_hj
import data_writetfrecord as datawrite
import data_utils


def generate_ode_auto_const(seed, eqns, quests, length, dt, num, caption_mode, name):
  '''du/dt = a * c + b'''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
  coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  all_ts = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
    for j in range(quests):
      ts_expand, control, traj = dyn.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_const_batch_fn, 
                                                      dt = dt, length = length, num = num,
                                                      k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                      coeffs = (coeff_a, coeff_b))
      all_ts.append(ts_expand)
      all_cs.append(control)
      all_us.append(traj)
      all_params.append("{:.8f}_{:.8f}".format(coeff_a, coeff_b))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ['forward', 'inverse']:
    datawrite.write_ode_tfrecord(name = name, eqn_type = "ode_auto_const", 
                      all_params = all_params, all_eqn_captions = all_eqn_captions,
                      all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                      problem_type = ptype)


def generate_ode_auto_linear1(seed, eqns, quests, length, dt, num, caption_mode, name):
  '''du/dt = a1 * c * u + a2'''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
  coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)

  all_ts = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)): 
    for j in range(quests):   
      ts_expand, control, traj = dyn.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear1_batch_fn, 
                                                      dt = dt, length = length, num = num,
                                                      k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                      coeffs = (coeff_a, coeff_b))
      all_ts.append(ts_expand)
      all_cs.append(control)
      all_us.append(traj)
      all_params.append("{:.8f}_{:.8f}".format(coeff_a, coeff_b))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ['forward', 'inverse']:
    datawrite.write_ode_tfrecord(name = name, eqn_type = "ode_auto_linear1", 
                        all_params = all_params, all_eqn_captions = all_eqn_captions,
                        all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                        problem_type = ptype)

def generate_ode_auto_linear2(seed, eqns, quests, length, dt, num, caption_mode, name):
  '''du/dt = a1 * u + a2 * c + a3'''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_a1 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeffs_a2 = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
  coeffs_a3 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  all_ts = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a1, coeff_a2, coeff_a3) in enumerate(zip(coeffs_a1, coeffs_a2, coeffs_a3)):
    for j in range(quests):
      ts_expand, control, traj = dyn.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear2_batch_fn, 
                                                      dt = dt, length = length, num = num,
                                                      k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                      coeffs = (coeff_a1, coeff_a2, coeff_a3))
      all_ts.append(ts_expand)
      all_cs.append(control)
      all_us.append(traj)
      all_params.append("{:.8f}_{:.8f}_{:.8f}".format(coeff_a1, coeff_a2, coeff_a3))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ['forward', 'inverse']:
    datawrite.write_ode_tfrecord(name = name, eqn_type = "ode_auto_linear2", 
                        all_params = all_params, all_eqn_captions = all_eqn_captions,
                        all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                        problem_type = ptype)


def generate_damped_oscillator(seed, eqns, quests, length, dt, num, caption_mode, name):
  '''
  damped_oscillator
  '''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  ts = jnp.arange(length*2) * dt/2
  ts_first = einshape("i->jik", ts[:length], j = num, k = 1) # (num, length, 1)
  ts_second = einshape("i->jik", ts[length:], j = num, k = 1) # (num, length, 1)
  decays = jax.random.uniform(next(rng), (eqns,), minval = 0.0, maxval = 2.0)
  all_ts_first = []; all_us_first = []; all_ts_second = []; all_us_second = []; all_params = []; all_eqn_captions = []
  for i, decay in enumerate(decays):
    for j in range(quests):
      amps = jax.random.uniform(next(rng), (num,), minval = 0.5, maxval = 1.5)
      periods = jax.random.uniform(next(rng), (num,), minval = 0.1, maxval = 0.2)
      phases = jax.random.uniform(next(rng), (num,), minval = 0.0, maxval = 2 * jnp.pi)
      time_series = series.generate_damped_oscillator_batch(ts, amps, periods, phases, decay)
      us_first = time_series[:, :length, None] # (num, length, 1)
      us_second = time_series[:, length:, None] # (num, length, 1)
      all_ts_first.append(ts_first)
      all_us_first.append(us_first)
      all_ts_second.append(ts_second)
      all_us_second.append(us_second)
      all_params.append("{:.8f}".format(decay))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ['forward', 'inverse']:
    datawrite.write_series_tfrecord(name = name, eqn_type = "series_damped_oscillator", 
                      all_params = all_params, all_eqn_captions = all_eqn_captions, 
                      all_ts_first = all_ts_first, all_us_first = all_us_first,
                      all_ts_second = all_ts_second, all_us_second = all_us_second,
                      problem_type = ptype)
  

def generate_pde_poisson(seed, eqns, quests, length, dx, num, caption_mode, name):
  '''
  du/dxx = c(x) over domain [0,L]
  c(x) : spatially varying coefficient, size N-1,
          we use GP to sample c(x)
  u_left, u_right: boundary conditions. 
  the output is the full solution, (N+1) grid point values.  
  '''
  N = length
  L = length * dx
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  all_xs = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_ul, coeff_ur) in enumerate(zip(coeffs_ul, coeffs_ur)):
    for j in range(quests):
      xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
      cs = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
      us = pdes.solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
      all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
      all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
      all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
      all_params.append("{:.8f}_{:.8f}".format(coeff_ul, coeff_ur))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ['forward', 'inverse']:
    datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_poisson_spatial", 
                      all_params = all_params, all_eqn_captions = all_eqn_captions,
                      all_xs = all_xs, all_ks = all_cs, all_us = all_us,
                      problem_type = ptype)



def generate_pde_porous(seed, eqns, quests, length, dx, num, caption_mode, name):
  '''
  - lamda * a * u_xx + k(x) u = c, a > 0, k > 0
  over domain [0,L]
  a, c : constants
  k(x) : spatially varying coefficient, size N-1,
          we use softplus(w), where w is generated using GP.
  u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
  the output is the full solution, (N+1) grid point values.  
  '''
  N = length
  L = length * dx
  lamda = 0.05
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeff_cs = jax.random.uniform(next(rng), (eqns,), minval = -2.0, maxval = 2.0)
  coeff_as = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
  
  all_xs = []; all_ks = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_ul, coeff_ur, coeff_c, coeff_a) in enumerate(zip(coeffs_ul, coeffs_ur, coeff_cs, coeff_as)):
    for j in range(quests):
      xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
      ks_GP = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) # (num, N+1)
      ks = jax.nn.softplus(ks_GP) # (num, N+1)
      us = pdes.solve_porous_batch(L, N, coeff_ul, coeff_ur, coeff_a * lamda, ks[:,1:-1], coeff_c) # (num, N+1)
      all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
      all_ks.append(einshape("ij->ijk", ks, k = 1)) # (num, N+1, 1)
      all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
      all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(coeff_ul, coeff_ur, coeff_c, coeff_a))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ['forward', 'inverse']:
    datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_porous_spatial", 
                      all_params = all_params, all_eqn_captions = all_eqn_captions,
                      all_xs = all_xs, all_ks = all_ks, all_us = all_us,
                      problem_type = ptype)


def generate_pde_cubic(seed, eqns, quests, length, dx, num, caption_mode, name):
  '''
  - a * lamda u_xx + k u^3 = c(x), a > 0, k > 0
  over domain [0,L]
  a, k : constants
  c(x) : spatially varying coefficient, size (N+1,)
  u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
  the output is the full solution and k, (N+1) grid point values.  
  u: a given profile (possibly a GP), need to be matched with u_left, u_right
    size [N+1,]
  '''
  N = length
  L = length * dx
  lamda = 0.1
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
  coeff_as = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
  coeff_ks = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
  all_xs = []; all_cs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_ul, coeff_ur, coeff_k, coeff_a) in enumerate(zip(coeffs_ul, coeffs_ur, coeff_ks, coeff_as)):
    for j in range(quests):
      xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
      us_GP = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) # (num, N+1)  
      [us,cs] = pdes.solve_cubic_batch(L, N, us_GP, coeff_ul, coeff_ur, coeff_a * lamda, coeff_k) 
      all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
      all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
      all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
      all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(coeff_ul, coeff_ur, coeff_a, coeff_k))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ["forward","inverse"]:
    datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_cubic_spatial",
                      all_params = all_params, all_eqn_captions = all_eqn_captions,
                      all_xs = all_xs, all_ks = all_cs, all_us = all_us,
                      problem_type = ptype)
      
def generate_mfc_gparam_hj(seed, eqns, quests, length, dx, dt, nu_nx_ratio, num, caption_mode, name):
  '''
  parameters: terminal cost g
  condition/qoi: initial rho and terminal rho
  '''
  run_cost = 20.0
  N = length # nx
  L = length * dx # should be 1.0
  assert L == 1.0
  terminal_t = 1.0
  Nt = int(terminal_t/dt)
  assert Nt * dt == terminal_t
  diffusion_eps = 0.04

  xs = jnp.linspace(0.0, 1.0, N, endpoint=False) #(nx,)
  xs_grids = einshape('i->ji', xs, j = Nt+1)  # (nt+1, nx)
  xs_grids = einshape('ij->(ij)', xs_grids)  # (n_pts), where n_pts = (nt+1) * nx
  ts_grids = einshape('i->ij', jnp.linspace(0, terminal_t, Nt+1), j = N)  # (nt+1, nx)
  ts_grids = einshape('ij->(ij)', ts_grids)  # (n_pts), where n_pts = (nt+1) * nx

  txs_grids = jnp.stack([ts_grids, xs_grids], axis = -1) # ((nt+1) * nx, 2)
  txs_grids_batch = einshape("...->j...", txs_grids, j = num) # (num, (nt+1) * nx, 2)

  nu = nu_nx_ratio * N
  us = jnp.linspace(0.0, 1.0, nu, endpoint=False) # (nu,)
  half_unroll_nums = mfc_hj.get_half_unroll_nums(us, us, ts_grids, terminal_t, diffusion_eps)
  
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  gs = data_utils.generate_gaussian_process(next(rng), us, eqns, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
  gs = gs - jnp.mean(gs, axis = -1, keepdims = True) # (eqns, nu)
  all_txs = []; all_rhos = []; all_params = []; all_eqn_captions = []
  for i, g in enumerate(gs):
    for j in range(quests):
      init_rho = data_utils.generate_gaussian_process(next(rng), us, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (num, nu)
      init_rho = jax.nn.softplus(init_rho) # (num, nu), positive
      init_rho = init_rho / jnp.mean(init_rho, axis = -1, keepdims = True) # (num, nu), sum to 1
      g_batch = einshape("i->ji", g, j = num) # (num, nu)
      rhos, _ = mfc_hj.solve_mfc_periodic_batch(g_batch/run_cost, us, init_rho, us, xs_grids, ts_grids, terminal_t, diffusion_eps, half_unroll_nums)
      all_txs.append(np.array(txs_grids_batch)) # (num, (nt+1) * nx, 2)
      all_rhos.append(np.array(rhos[...,None])) # (num, (nt+1) * nx, 1)
      # 10 data points, 0, 1/10, 2/10, ..., 9/10
      all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}"
                    .format(g[0], g[nu//10], g[2*nu//10], g[3*nu//10], g[4*nu//10], g[5*nu//10], g[6*nu//10], g[7*nu//10], g[8*nu//10], g[9*nu//10]))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ["forward11", "forward12", "forward22"]:
    datawrite.write_mfc_gparam_hj_tfrecord(name = name, eqn_type = "mfc_gparam_hj",
                                            all_params = all_params, all_eqn_captions = all_eqn_captions,
                                            all_txs = all_txs, all_rhos = all_rhos,
                                            problem_type = ptype, nt = Nt+1, nx = N)

def generate_mfc_rhoparam_hj(seed, eqns, quests, length, dx, dt, nu_nx_ratio, num, caption_mode, name):
  '''
  parameters: terminal cost g
  condition/qoi: initial rho and terminal rho
  '''
  run_cost = 20.0
  N = length # nx
  L = length * dx # should be 1.0
  assert L == 1.0
  terminal_t = 1.0
  Nt = int(terminal_t/dt)
  assert Nt * dt == terminal_t
  diffusion_eps = 0.04

  xs = jnp.linspace(0.0, 1.0, N, endpoint=False) #(nx,)
  xs_grids = einshape('i->ji', xs, j = Nt+1)  # (nt+1, nx)
  xs_grids = einshape('ij->(ij)', xs_grids)  # (n_pts), where n_pts = (nt+1) * nx
  ts_grids = einshape('i->ij', jnp.linspace(0, terminal_t, Nt+1), j = N)  # (nt+1, nx)
  ts_grids = einshape('ij->(ij)', ts_grids)  # (n_pts), where n_pts = (nt+1) * nx

  txs_grids = jnp.stack([ts_grids, xs_grids], axis = -1) # ((nt+1) * nx, 2)
  txs_grids_batch = einshape("...->j...", txs_grids, j = num) # (num, (nt+1) * nx, 2)

  nu = nu_nx_ratio * N
  us = jnp.linspace(0.0, 1.0, nu, endpoint=False) # (nu,)
  half_unroll_nums = mfc_hj.get_half_unroll_nums(us, us, ts_grids, terminal_t, diffusion_eps)

  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  init_rhos = data_utils.generate_gaussian_process(next(rng), us, eqns, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (eqns, nu)
  init_rhos = jax.nn.softplus(init_rhos) # (eqns, nu), positive
  init_rhos = init_rhos / jnp.mean(init_rhos, axis = -1, keepdims = True) # (eqns, nu)
  all_rhos_key = []; all_rhos_value = []; all_gs_key = []; all_gs_value = []; all_params = []; all_eqn_captions = []
  for i, init_rho in enumerate(init_rhos):
    for j in range(quests):
      g_batch = data_utils.generate_gaussian_process(next(rng), us, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
      g_batch = g_batch - jnp.mean(g_batch, axis = -1, keepdims = True) # (num, nu)
      init_rho_batch = einshape("i->ji", init_rho, j = num) # (num, nu)
      rhos, _ = mfc_hj.solve_mfc_periodic_batch(g_batch/run_cost, us, init_rho_batch, us, xs_grids, ts_grids, terminal_t, diffusion_eps, half_unroll_nums)
      all_rhos_key.append(np.array(txs_grids_batch)) # (num, (nt+1) * nx, 2)
      all_rhos_value.append(np.array(rhos[...,None])) # (num, (nt+1) * nx, 1)
      all_gs_key.append(np.array(einshape("i->jik", xs, j = num, k = 1))) # (num, nx, 1)
      all_gs_value.append(np.array(g_batch[:,::nu_nx_ratio,None])) # (num, nx, 1)
      # 10 data points, 0, 1/10, 2/10, ..., 9/10
      all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}"
                    .format(init_rho[0], init_rho[nu//10], init_rho[2*nu//10], init_rho[3*nu//10], init_rho[4*nu//10], init_rho[5*nu//10], init_rho[6*nu//10], init_rho[7*nu//10], init_rho[8*nu//10], init_rho[9*nu//10]))
      all_eqn_captions.append(None)
    utils.print_dot(i)
  for ptype in ["forward11", "forward12"]:
    datawrite.write_mfc_rhoparam_hj_tfrecord(name = name, eqn_type = "mfc_rhoparam_hj", 
                                              all_params = all_params, all_eqn_captions = all_eqn_captions,
                                              all_rhos_key = all_rhos_key, all_rhos_value = all_rhos_value, 
                                              all_gs_key = all_gs_key, all_gs_value = all_gs_value,
                                              problem_type = ptype, nt = Nt+1, nx = N)



def main(argv):
  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)
  
  
  name = '{}/{}'.format(FLAGS.dir, FLAGS.name)

  if not os.path.exists(FLAGS.dir):
    os.makedirs(FLAGS.dir)
    
  if 'ode_auto_const' in FLAGS.eqn_types:
    generate_ode_auto_const(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
                            dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
    
  if 'ode_auto_linear1' in FLAGS.eqn_types:
    generate_ode_auto_linear1(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
                              dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
    
  if 'ode_auto_linear2' in FLAGS.eqn_types:
    generate_ode_auto_linear2(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
                              dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
    
  if 'series_damped_oscillator' in FLAGS.eqn_types:
    generate_damped_oscillator(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
                              dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
    
  if 'pde_poisson_spatial' in FLAGS.eqn_types:
    generate_pde_poisson(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
                              dx = FLAGS.dx, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
    
  if 'pde_porous_spatial' in FLAGS.eqn_types:
    generate_pde_porous(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length,
                        dx = FLAGS.dx, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)


  if 'pde_cubic_spatial' in FLAGS.eqn_types: 
    generate_pde_cubic(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length,
                        dx = FLAGS.dx, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
  
  if 'mfc_gparam_hj' in FLAGS.eqn_types:
    generate_mfc_gparam_hj(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length,
                          dx = FLAGS.dx, dt = FLAGS.dt, nu_nx_ratio = FLAGS.nu_nx_ratio,
                          num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
  
  if 'mfc_rhoparam_hj' in FLAGS.eqn_types:
    generate_mfc_rhoparam_hj(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length,
                          dx = FLAGS.dx, dt = FLAGS.dt, nu_nx_ratio = FLAGS.nu_nx_ratio,
                          num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name)
    
if __name__ == "__main__":

  import tensorflow as tf
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  tf.config.set_visible_devices([], device_type='GPU')

  FLAGS = flags.FLAGS
  flags.DEFINE_string('caption_mode', None, 'mode for caption')
  flags.DEFINE_integer('num', 100, 'number of systems in each equation')
  flags.DEFINE_integer('quests', 1, 'number of questions in each operator')
  flags.DEFINE_integer('eqns', 1000, 'number of equations')
  flags.DEFINE_integer('length', 50, 'length of trajectory and control')
  flags.DEFINE_integer('mfc_iters', 1000, 'iterations for solving mfc')
  flags.DEFINE_float('mfc_tol', 1e-10, 'res tolerance for solving mfc')
  flags.DEFINE_boolean('mfc_verbose', False, 'verbose for solving mfc')
  flags.DEFINE_float('dt', 0.02, 'time step in dynamics')
  flags.DEFINE_float('dx', 0.02, 'time step in dynamics')
  flags.DEFINE_integer('nu_nx_ratio', 1, 'nu_nx_ratio in mfc_hj')
  flags.DEFINE_string('name', 'data', 'name of the dataset')
  flags.DEFINE_string('dir', '.', 'name of the directory to save the data')
  flags.DEFINE_list('eqn_types', [], 'list of equations for data generation')
  flags.DEFINE_list('write', [], 'list of features to write')

  flags.DEFINE_integer('seed', 1, 'random seed')

  app.run(main)
