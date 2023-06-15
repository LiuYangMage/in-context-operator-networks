import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
from scipy.spatial.distance import cdist
import pickle
from functools import partial
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

# Define the covariance function
def rbf_kernel(x1, x2, sigma, l):
    """
    Radial basis function kernel
    """
    sq_norm = cdist(x1 / l, x2 / l, metric='sqeuclidean')
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_kernel_jax(x1, x2, sigma, l):
    """
    Radial basis function kernel, only support 1D x1 and x2
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1-xx2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_sin_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (jnp.sin(jnp.pi*(xx1-xx2)))**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

def rbf_circle_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    xx1_1 = jnp.sin(xx1 * 2 * jnp.pi)
    xx1_2 = jnp.cos(xx1 * 2 * jnp.pi)
    xx2_1 = jnp.sin(xx2 * 2 * jnp.pi)
    xx2_2 = jnp.cos(xx2 * 2 * jnp.pi)
    sq_norm = (xx1_1-xx2_1)**2/(l**2) + (xx1_2-xx2_2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

@partial(jax.jit, static_argnames=('num','kernel'))
def generate_gaussian_process(key, ts, num, kernel, k_sigma, k_l):
  '''
  ts: 1D array (length,)
  out: Gaussian process samples, 2D array (num, length)
  '''
  length = len(ts)
  mean = jnp.zeros((num,length))
  # cov = rbf_kernel(ts[:, None], ts[:, None], sigma=k_sigma, l=k_l)
  cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
  cov = einshape('ii->nii', cov, n = num)
  out = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
  return out

@partial(jax.jit, static_argnames=('ode_batch_fn','length','num',))
def generate_one_dyn(key, ode_batch_fn, dt, length, num, k_sigma, k_l, init_range, coeffs,
                     control = None):
  '''
  generate data for dynamics
  @param 
    key: jax.random.PRNGKey
    ode_batch_fn: e.g. ode_auto_const_batch_fn, jitted function
    dt: float, time step
    length: int, length of time series
    num: int, number of samples
    k_sigma, k_l: float, kernel parameters
    init_range: tuple, range of initial values
    coeffs: tuple, coefficients of the dynamics, will be unpacked and passed to ode_batch_fn
    control: 2D array (num, length), control signal, if None, generate with Gaussian process
  @return
    ts: 2D array (num, length, 1), time series
    control: 2D array (num, length, 1), control signal
    traj: 2D array (num, length, 1), trajectory
  '''
  ts = jnp.arange(length) * dt
  key, subkey1, subkey2 = jax.random.split(key, num = 3)
  if control is None:
    control = generate_gaussian_process(subkey1, ts, num, kernel = rbf_kernel_jax, k_sigma = k_sigma, k_l = k_l)
  init = jax.random.uniform(subkey2, (num,), minval = init_range[0], maxval = init_range[1])
  # traj[0] = init, final is affected by control[-1]
  _, traj = ode_batch_fn(init, control, dt, *coeffs)
  ts_expand = einshape("i->ji", ts, j = num)
  return ts_expand[...,None], control[...,None], traj[...,None]

def main(argv):
  import data_dynamics as dyn
  import data_series as series
  import data_pdes as pdes
  import data_mfc_hj as mfc_hj
  import datawrite_tfrecord as datawrite

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)
  
  length = FLAGS.length
  dt = FLAGS.dt
  dx = FLAGS.dx
  num = FLAGS.num
  eqns = FLAGS.eqns
  quests = FLAGS.quests

  if not os.path.exists(FLAGS.dir):
    os.makedirs(FLAGS.dir)
    
  if 'ode_auto_const' in FLAGS.eqn_types:
    '''du/dt = a * c + b'''
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    all_ts = []; all_cs = []; all_us = []; all_params = []
    for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
      for j in range(quests):
        ts_expand, control, traj = generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_const_batch_fn, 
                                                    dt = dt, length = length, num = num,
                                                    k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                    coeffs = (coeff_a, coeff_b))
        all_ts.append(ts_expand)
        all_cs.append(control)
        all_us.append(traj)
        all_params.append("{:.3f}_{:.3f}".format(coeff_a, coeff_b))
      if i % 100 == 0:
        print(i, all_params[-1], flush = True)

    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_ode_tfrecord(name = name, eqn_type = "ode_auto_const", 
                       all_params = all_params,
                       all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                       problem_types = ['forward', 'inverse'])

  if 'ode_auto_linear1' in FLAGS.eqn_types:
    '''du/dt = a1 * c * u + a2'''
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)

    all_ts = []; all_cs = []; all_us = []; all_params = []
    for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)): 
      for j in range(quests):   
        ts_expand, control, traj = generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear1_batch_fn, 
                                                    dt = dt, length = length, num = num,
                                                    k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                    coeffs = (coeff_a, coeff_b))
        all_ts.append(ts_expand)
        all_cs.append(control)
        all_us.append(traj)
        all_params.append("{:.3f}_{:.3f}".format(coeff_a, coeff_b))
      if i % 100 == 0:
        print(i, all_params[-1], flush = True)

    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_ode_tfrecord(name = name, eqn_type = "ode_auto_linear1", 
                       all_params = all_params,
                       all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                       problem_types = ['forward', 'inverse'])

  if 'ode_auto_linear2' in FLAGS.eqn_types:
    '''du/dt = a1 * u + a2 * c + a3'''
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_a1 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    coeffs_a2 = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    coeffs_a3 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    all_ts = []; all_cs = []; all_us = []; all_params = []
    for i, (coeff_a1, coeff_a2, coeff_a3) in enumerate(zip(coeffs_a1, coeffs_a2, coeffs_a3)):
      for j in range(quests):
        ts_expand, control, traj = generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear2_batch_fn, 
                                                    dt = dt, length = length, num = num,
                                                    k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                    coeffs = (coeff_a1, coeff_a2, coeff_a3))
        all_ts.append(ts_expand)
        all_cs.append(control)
        all_us.append(traj)
        all_params.append("{:.3f}_{:.3f}_{:.3f}".format(coeff_a1, coeff_a2, coeff_a3))
      if i % 100 == 0:
        print(i, all_params[-1], flush = True)
    
    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_ode_tfrecord(name = name, eqn_type = "ode_auto_linear2", 
                       all_params = all_params,
                       all_ts = all_ts, all_cs = all_cs, all_us = all_us,
                       problem_types = ['forward', 'inverse'])
  

  if 'series_damped_oscillator' in FLAGS.eqn_types:
    '''
    damped_oscillator
    '''
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    ts = jnp.arange(length*2) * dt/2
    ts_first = einshape("i->jik", ts[:length], j = num, k = 1) # (num, length, 1)
    ts_second = einshape("i->jik", ts[length:], j = num, k = 1) # (num, length, 1)
    decays = jax.random.uniform(next(rng), (eqns,), minval = 0.0, maxval = 2.0)
    all_ts_first = []; all_us_first = []; all_ts_second = []; all_us_second = []; all_params = []
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
        all_params.append("{:.3f}".format(decay))
      if i % 100 == 0:
        print(i, flush = True)  

    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_series_tfrecord(name = name, eqn_type = "series_damped_oscillator", 
                      all_params = all_params,  
                      all_ts_first = all_ts_first, all_us_first = all_us_first,
                      all_ts_second = all_ts_second, all_us_second = all_us_second,
                      problem_types = ["forward","inverse"])
  
  if 'pde_poisson_spatial' in FLAGS.eqn_types:
    '''
    du/dxx = c(x) over domain [0,L]
    c(x) : spatially varying coefficient, size N-1,
           we use GP to sample c(x)
    u_left, u_right: boundary conditions. 
    the output is the full solution, (N+1) grid point values.  
    '''
    N = length
    L = length * dx
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    all_xs = []; all_cs = []; all_us = []; all_params = []
    for i, (coeff_ul, coeff_ur) in enumerate(zip(coeffs_ul, coeffs_ur)):
      for j in range(quests):
        xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
        cs = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 2.0, k_l = 0.5) # (num, N+1)
        us = pdes.solve_poisson_batch(L, N, coeff_ul, coeff_ur, cs[:,1:-1]) # (num, N+1)
        all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
        all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
        all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
        all_params.append("{:.3f}_{:.3f}".format(coeff_ul, coeff_ur))
      if i % 100 == 0:
        print(i, flush = True)     
    
    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_poisson_spatial", 
                      all_params = all_params,  
                      all_xs = all_xs, all_ks = all_cs, all_us = all_us,
                      problem_types = ["forward","inverse"])


  if 'pde_porous_spatial' in FLAGS.eqn_types:
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
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    coeff_cs = jax.random.uniform(next(rng), (eqns,), minval = -2.0, maxval = 2.0)
    coeff_as = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    
    all_xs = []; all_ks = []; all_us = []; all_params = []
    for i, (coeff_ul, coeff_ur, coeff_c, coeff_a) in enumerate(zip(coeffs_ul, coeffs_ur, coeff_cs, coeff_as)):
      for j in range(quests):
        xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
        ks_GP = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) # (num, N+1)
        ks = jax.nn.softplus(ks_GP) # (num, N+1)
        us = pdes.solve_porous_batch(L, N, coeff_ul, coeff_ur, coeff_a * lamda, ks[:,1:-1], coeff_c) # (num, N+1)
        all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
        all_ks.append(einshape("ij->ijk", ks, k = 1)) # (num, N+1, 1)
        all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
        all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(coeff_ul, coeff_ur, coeff_c, coeff_a))
      if i % 100 == 0:
        print(i, flush = True)  

    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_porous_spatial", 
                      all_params = all_params,  
                      all_xs = all_xs, all_ks = all_ks, all_us = all_us,
                      problem_types = ["forward","inverse"])
  

  if 'pde_cubic_spatial' in FLAGS.eqn_types: 
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
      rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
      coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
      coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
      coeff_as = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
      coeff_ks = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
      all_xs = []; all_cs = []; all_us = []; all_params = []
      for i, (coeff_ul, coeff_ur, coeff_k, coeff_a) in enumerate(zip(coeffs_ul, coeffs_ur, coeff_ks, coeff_as)):
        for j in range(quests):
          xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
          us_GP = generate_gaussian_process(next(rng), xs, num, kernel = rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) # (num, N+1)  
          [us,cs] = pdes.solve_cubic_batch(L, N, us_GP, coeff_ul, coeff_ur, coeff_a * lamda, coeff_k) 
          all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
          all_cs.append(einshape("ij->ijk", cs, k = 1)) # (num, N+1, 1)
          all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
          all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(coeff_ul, coeff_ur, coeff_a, coeff_k))
        if i % 100 == 0:
          print(i, flush = True) 

      name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
      datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_cubic_spatial",
                        all_params = all_params,
                        all_xs = all_xs, all_ks = all_cs, all_us = all_us,
                        problem_types = ["forward","inverse"])
      

  if 'mfc_gparam_hj' in FLAGS.eqn_types:
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

    nu_nx_ratio = FLAGS.nu_nx_ratio
    nu = nu_nx_ratio * N
    us = jnp.linspace(0.0, 1.0, nu, endpoint=False) # (nu,)
    half_unroll_nums = mfc_hj.get_half_unroll_nums(us, us, ts_grids, terminal_t, diffusion_eps)
    
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    gs = generate_gaussian_process(next(rng), us, eqns, kernel = rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
    gs = gs - jnp.mean(gs, axis = -1, keepdims = True) # (eqns, nu)
    all_txs = []; all_rhos = []; all_params = []
    for i, g in enumerate(gs):
      for j in range(quests):
        init_rho = generate_gaussian_process(next(rng), us, num, kernel = rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (num, nu)
        init_rho = jax.nn.softplus(init_rho) # (num, nu), positive
        init_rho = init_rho / jnp.mean(init_rho, axis = -1, keepdims = True) # (num, nu), sum to 1
        g_batch = einshape("i->ji", g, j = num) # (num, nu)
        rhos, _ = mfc_hj.solve_mfc_periodic_batch(g_batch/run_cost, us, init_rho, us, xs_grids, ts_grids, terminal_t, diffusion_eps, half_unroll_nums)
        all_txs.append(np.array(txs_grids_batch)) # (num, (nt+1) * nx, 2)
        all_rhos.append(np.array(rhos[...,None])) # (num, (nt+1) * nx, 1)
        all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(g[0], g[nu//3], g[2*nu//3], g[-1]))
      if i % 100 == 0:
        print(""); print(i, end = "", flush=True)
      print(".", end = "", flush = True)
  
    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_mfc_gparam_hj_tfrecord(name = name, eqn_type = "mfc_gparam_hj",
                                all_params = all_params, all_txs = all_txs, all_rhos = all_rhos,
                                problem_types = ["forward11", "forward12", "forward22"], nt = Nt+1, nx = N)


  if 'mfc_rhoparam_hj' in FLAGS.eqn_types:
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

    nu_nx_ratio = FLAGS.nu_nx_ratio
    nu = nu_nx_ratio * N
    us = jnp.linspace(0.0, 1.0, nu, endpoint=False) # (nu,)
    half_unroll_nums = mfc_hj.get_half_unroll_nums(us, us, ts_grids, terminal_t, diffusion_eps)

    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    init_rhos = generate_gaussian_process(next(rng), us, eqns, kernel = rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (eqns, nu)
    init_rhos = jax.nn.softplus(init_rhos) # (eqns, nu), positive
    init_rhos = init_rhos / jnp.mean(init_rhos, axis = -1, keepdims = True) # (eqns, nu)
    all_rhos_key = []; all_rhos_value = []; all_gs_key = []; all_gs_value = []; all_params = []
    for i, init_rho in enumerate(init_rhos):
      for j in range(quests):
        g_batch = generate_gaussian_process(next(rng), us, num, kernel = rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
        g_batch = g_batch - jnp.mean(g_batch, axis = -1, keepdims = True) # (num, nu)
        init_rho_batch = einshape("i->ji", init_rho, j = num) # (num, nu)
        rhos, _ = mfc_hj.solve_mfc_periodic_batch(g_batch/run_cost, us, init_rho_batch, us, xs_grids, ts_grids, terminal_t, diffusion_eps, half_unroll_nums)
        all_rhos_key.append(np.array(txs_grids_batch)) # (num, (nt+1) * nx, 2)
        all_rhos_value.append(np.array(rhos[...,None])) # (num, (nt+1) * nx, 1)
        all_gs_key.append(np.array(einshape("i->jik", xs, j = num, k = 1))) # (num, nx, 1)
        all_gs_value.append(np.array(g_batch[:,::nu_nx_ratio,None])) # (num, nx, 1)
        all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(init_rho[0], init_rho[nu//3], init_rho[2*nu//3], init_rho[-1]))
      if i % 100 == 0:
        print(""); print(i, end = "", flush=True)
      print(".", end = "", flush = True)
  
    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_mfc_rhoparam_hj_tfrecord(name = name, eqn_type = "mfc_rhoparam_hj", all_params = all_params, 
                                all_rhos_key = all_rhos_key, all_rhos_value = all_rhos_value, 
                                all_gs_key = all_gs_key, all_gs_value = all_gs_value,
                                problem_types = ["forward11", "forward12"], nt = Nt+1, nx = N)


if __name__ == "__main__":

  import tensorflow as tf
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  tf.config.set_visible_devices([], device_type='GPU')

  FLAGS = flags.FLAGS
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

  flags.DEFINE_integer('seed', 1, 'random seed')

  app.run(main)
