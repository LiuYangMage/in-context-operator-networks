import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
from scipy.spatial.distance import cdist
import pickle
from functools import partial
import tensorflow as tf
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk

import data_dynamics as dyn
import data_series as series
import data_pdes as pdes
import datagen
import datawrite_tfrecord as datawrite

import data_mfc_hj as mfc_hj

tf.config.set_visible_devices([], device_type='GPU')

def main(argv):
  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)
  
  length = FLAGS.length
  dx = FLAGS.dx
  dt = FLAGS.dt
  num = FLAGS.num
  eqns = FLAGS.eqns
  quests = FLAGS.quests


  if not os.path.exists(FLAGS.dir):
    os.makedirs(FLAGS.dir)


  if 'ode_auto_const' in FLAGS.eqn_types:
    '''du/dt = a * c + b'''
    utils.timer.tic('ode_auto_const')
    # training distribution
    # coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    # coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
    bmins = np.linspace(-3, 3, FLAGS.ood_coeff2_grids)[:-1]; gap_b = bmins[1] - bmins[0]
    print("amin", amins, flush = True)
    print("bmin", bmins, flush = True)
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    ts = jnp.arange(length) * dt
    controls = [[datagen.generate_gaussian_process(next(rng), ts, num, kernel = datagen.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) 
                for j in range(quests)] for i in range(eqns)]
    for a_min in amins:
      for b_min in bmins:
        rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed)) #reset demo
        coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = a_min, maxval = a_min + gap_a)
        coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = b_min, maxval = b_min + gap_b)
        all_ts = []; all_cs = []; all_us = []; all_params = []
        for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
          for j in range(quests):
            ts_expand, control_expand, traj = datagen.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_const_batch_fn, 
                                                              dt = dt, length = length, num = num,
                                                              k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                              coeffs = (coeff_a, coeff_b), control = controls[i][j])
            all_ts.append(ts_expand)
            all_cs.append(control_expand)
            all_us.append(traj)
            all_params.append("{:.3f}_{:.3f}".format(coeff_a, coeff_b))
          if i % 100 == 0:
            print(i, all_params[-1], flush = True)
        name = '{}/{}_{:.2f}_{:.2f}'.format(FLAGS.dir, FLAGS.name, a_min, b_min)
        datawrite.write_ode_tfrecord(name, 'ode_auto_const', all_params, all_ts, all_cs, all_us, ['forward', 'inverse'])

    utils.timer.toc('ode_auto_const')

  if 'ode_auto_linear1' in FLAGS.eqn_types:
    '''du/dt = a * c * u + b'''
    utils.timer.tic('ode_auto_linear1')
    # training distribution
    # coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    # coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
    bmins = np.linspace(-3, 3, FLAGS.ood_coeff2_grids)[:-1]; gap_b = bmins[1] - bmins[0]
    print("amin", amins, flush = True)
    print("bmin", bmins, flush = True)
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    ts = jnp.arange(length) * dt
    controls = [[datagen.generate_gaussian_process(next(rng), ts, num, kernel = datagen.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) 
                for j in range(quests)] for i in range(eqns)]
    for a_min in amins:
      for b_min in bmins:
        rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed)) #reset demo
        coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = a_min, maxval = a_min + gap_a)
        coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = b_min, maxval = b_min + gap_b)
        all_ts = []; all_cs = []; all_us = []; all_params = []
        for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
          for j in range(quests):
            ts_expand, control_expand, traj = datagen.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear1_batch_fn, 
                                                              dt = dt, length = length, num = num,
                                                              k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                              coeffs = (coeff_a, coeff_b), control = controls[i][j])
            all_ts.append(ts_expand)
            all_cs.append(control_expand)
            all_us.append(traj)
            all_params.append("{:.3f}_{:.3f}".format(coeff_a, coeff_b))
          if i % 100 == 0:
            print(i, all_params[-1], flush = True)
        name = '{}/{}_{:.2f}_{:.2f}'.format(FLAGS.dir, FLAGS.name, a_min, b_min)
        datawrite.write_ode_tfrecord(name, 'ode_auto_linear1', all_params, all_ts, all_cs, all_us, ['forward', 'inverse'])
    utils.timer.toc('ode_auto_linear1')

  if 'ode_auto_linear2' in FLAGS.eqn_types:
    '''du/dt = a1 * u + a2 * c + a3'''
    utils.timer.tic('ode_auto_linear2')
    # train distribution
    # coeffs_a1 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    # coeffs_a2 = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    # coeffs_a3 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    # here we make grids on a2 and a3, denote as a and b respectively
    amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
    bmins = np.linspace(-3, 3, FLAGS.ood_coeff2_grids)[:-1]; gap_b = bmins[1] - bmins[0]
    print("amin", amins, flush = True)
    print("bmin", bmins, flush = True)
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_a1 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
    ts = jnp.arange(length) * dt
    controls = [[datagen.generate_gaussian_process(next(rng), ts, num, kernel = datagen.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) 
                for j in range(quests)] for i in range(eqns)]
    for a_min in amins:
      for b_min in bmins:
        rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed)) #reset demo
        coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = a_min, maxval = a_min + gap_a)
        coeffs_b = jax.random.uniform(next(rng), (eqns,), minval = b_min, maxval = b_min + gap_b)
        all_ts = []; all_cs = []; all_us = []; all_params = []
        for i, (coeff_a1, coeff_a, coeff_b) in enumerate(zip(coeffs_a1, coeffs_a, coeffs_b)):
          for j in range(quests):
            ts_expand, control_expand, traj = datagen.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear2_batch_fn, 
                                                              dt = dt, length = length, num = num,
                                                              k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                              coeffs = (coeff_a1, coeff_a, coeff_b), control = controls[i][j])
            all_ts.append(ts_expand)
            all_cs.append(control_expand)
            all_us.append(traj)
            all_params.append("{:.3f}_{:.3f}_{:.3f}".format(coeff_a1, coeff_a, coeff_b))
          if i % 100 == 0:
            print(i, all_params[-1], flush = True)
        name = '{}/{}_{:.2f}_{:.2f}'.format(FLAGS.dir, FLAGS.name, a_min, b_min)
        datawrite.write_ode_tfrecord(name, 'ode_auto_linear2', all_params, all_ts, all_cs, all_us, ['forward', 'inverse'])
    utils.timer.toc('ode_auto_linear2')
    
  if 'pde_porous_spatial' in FLAGS.eqn_types:
    '''
    - lamda * a * u_xx + k(x) u = c, a > 0, k > 0
    over domain [0,L]
    parameters:
    a, c : constants
    u_left: left boundary condition sampled from [pde_bdry_left_min, pde_bdry_left_max] 
    u_right: right bdry sampled from [pde_bdry_right_min, pde_bdry_right_max]
    cond and qoi:
    k(x) : spatially varying coefficient, size N-1,
           we use softplus(w), where w is generated using GP.    
    the output is the full solution, (N+1) grid point values.  
    '''
    utils.timer.tic('pde_porous_spatial')
    N = length
    L = length * dx
    lamda = 0.05
    # training distribution
    # coeff_cs = jax.random.uniform(next(rng), (eqns,), minval = -2.0, maxval = 2.0)
    # coeff_as = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
    # NOTE: c is coeff1 and a is coeff2
    amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
    cmins = np.linspace(-6, 6, FLAGS.ood_coeff2_grids)[:-1]; gap_c = cmins[1] - cmins[0]
    print("cmin", cmins, flush = True)
    print("amin", amins, flush = True)
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    coeffs_ul = jax.random.uniform(next(rng), (eqns,), minval = FLAGS.pde_bdry_left_min, maxval = FLAGS.pde_bdry_left_max)
    coeffs_ur = jax.random.uniform(next(rng), (eqns,), minval = FLAGS.pde_bdry_right_min, maxval = FLAGS.pde_bdry_right_max)
    xs = jnp.linspace(0.0, 1.0, N+1)# (N+1,)
    ks_GPs = [[datagen.generate_gaussian_process(next(rng), xs, num, kernel = datagen.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) 
                for j in range(quests)] for i in range(eqns)]
    for a_min in amins:
      for c_min in cmins:
        rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed)) #reset demo
        coeffs_c = jax.random.uniform(next(rng), (eqns,), minval = c_min, maxval = c_min + gap_c)
        coeffs_a = jax.random.uniform(next(rng), (eqns,), minval = a_min, maxval = a_min + gap_a)
        all_xs = []; all_ks = []; all_us = []; all_params = []
        for i, (coeff_c, coeff_a, coeff_ul, coeff_ur) in enumerate(zip(coeffs_c, coeffs_a, coeffs_ul, coeffs_ur)):
          for j in range(quests):
            ks = jax.nn.softplus(ks_GPs[i][j]) # (num, N+1)
            us = pdes.solve_porous_batch(L, N, coeff_ul, coeff_ur, coeff_a * lamda, ks[:,1:-1], coeff_c) # (num, N+1)
            all_xs.append(einshape("i->jik", xs, j = num, k = 1)) # (num, N+1, 1)
            all_ks.append(einshape("ij->ijk", ks, k = 1)) # (num, N+1, 1)
            all_us.append(einshape("ij->ijk", us, k = 1)) # (num, N+1, 1)
            all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(coeff_ul, coeff_ur, coeff_c, coeff_a))
          if i % 100 == 0:
            print(i, all_params[-1], flush = True)
        name = '{}/{}_{:.2f}_{:.2f}'.format(FLAGS.dir, FLAGS.name, c_min, a_min)
        datawrite.write_pde_tfrecord(name = name, eqn_type = "pde_porous_spatial", 
                      all_params = all_params,  
                      all_xs = all_xs, all_ks = all_ks, all_us = all_us,
                      problem_types = ["forward","inverse"])
    utils.timer.toc('pde_porous_spatial')

  if 'series_damped_oscillator' in FLAGS.eqn_types:
    '''
    damped_oscillator
    '''
    utils.timer.tic('series_damped_oscillator')
    ts = jnp.arange(length*2) * dt/2
    ts_first = einshape("i->jik", ts[:length], j = num, k = 1) # (num, length, 1)
    ts_second = einshape("i->jik", ts[length:], j = num, k = 1) # (num, length, 1)
    # training distribution
    # decays = jax.random.uniform(next(rng), (eqns,), minval = 0.0, maxval = 2.0)
    decays_grids = np.linspace(-1.0, 5.0, FLAGS.ood_coeff1_grids)[:-1]; gap = decays_grids[1] - decays_grids[0]
    print('decays_grids', decays_grids, flush = True)
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    amps = [[jax.random.uniform(next(rng), (num,), minval = 0.5, maxval = 1.5) 
                for j in range(quests)] for i in range(eqns)]
    periods = [[jax.random.uniform(next(rng), (num,), minval = 0.1, maxval = 0.2)
                for j in range(quests)] for i in range(eqns)]
    phases = [[jax.random.uniform(next(rng), (num,), minval = 0.0, maxval = 2 * jnp.pi)
                for j in range(quests)] for i in range(eqns)]
    for decays_grid in decays_grids:
      rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed)) #reset demo
      decays = jax.random.uniform(next(rng), (eqns,), minval = decays_grid, maxval = decays_grid + gap)
      all_ts_first = []; all_us_first = []; all_ts_second = []; all_us_second = []; all_params = []
      for i, decay in enumerate(decays):
        for j in range(quests):
          time_series = series.generate_damped_oscillator_batch(ts, amps[i][j], periods[i][j], phases[i][j], decay)
          us_first = time_series[:, :length, None] # (num, length, 1)
          us_second = time_series[:, length:, None] # (num, length, 1)
          all_ts_first.append(ts_first)
          all_us_first.append(us_first)
          all_ts_second.append(ts_second)
          all_us_second.append(us_second)
          all_params.append("{:.3f}".format(decay))
        if i % 100 == 0:
          print(i, all_params[-1], flush = True)  
      name = '{}/{}_{:.2f}'.format(FLAGS.dir, FLAGS.name, decays_grid)
      datawrite.write_series_tfrecord(name = name, eqn_type = "series_damped_oscillator", 
                        all_params = all_params,  
                        all_ts_first = all_ts_first, all_us_first = all_us_first,
                        all_ts_second = all_ts_second, all_us_second = all_us_second,
                        problem_types = ["forward","inverse"])
    utils.timer.toc('series_damped_oscillator')

    
  # nt
  if 'ode_auto_linear3' in FLAGS.eqn_types:
    '''du/dt = a1 * c * u + a2 * u + a3'''
    utils.timer.tic('ode_auto_linear3')
    coeffs_a2 = jnp.linspace(-1.0,1.0,FLAGS.ood_coeff2_grids,endpoint=True)
    rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed))
    ts = jnp.arange(length) * dt
    controls = [[datagen.generate_gaussian_process(next(rng), ts, num, kernel = datagen.rbf_kernel_jax, k_sigma = 1.0, k_l = 0.5) 
                for j in range(quests)] for i in range(eqns)]
    for coeff_a2 in coeffs_a2:
      rng = hk.PRNGSequence(jax.random.PRNGKey(FLAGS.seed)) # reset random seed
      coeffs_a1 = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.5)
      coeffs_a3 = jax.random.uniform(next(rng), (eqns,), minval = -1.0, maxval = 1.0)
      all_ts = []; all_cs = []; all_us = []; all_params = []
      for i, (coeff_a1, coeff_a3) in enumerate(zip(coeffs_a1, coeffs_a3)):
        for j in range(quests):
          ts_expand, control_expand, traj = datagen.generate_one_dyn(key = next(rng), ode_batch_fn = dyn.ode_auto_linear3_batch_fn,
                                                            dt = dt, length = length, num = num,
                                                            k_sigma = 1.0, k_l = 0.5, init_range = (-1,1),
                                                            coeffs = (coeff_a1, coeff_a2, coeff_a3), control = controls[i][j])
          all_ts.append(ts_expand)
          all_cs.append(control_expand)
          all_us.append(traj)
          all_params.append("{:.8f}_{:.8f}_{:.8f}".format(coeff_a1, coeff_a2, coeff_a3))
        if i % 100 == 0:
          print(i, all_params[-1], flush = True)
      name = '{}/{}_{:.2f}'.format(FLAGS.dir, FLAGS.name, coeff_a2)
      datawrite.write_ode_tfrecord(name, 'ode_auto_linear3', all_params, all_ts, all_cs, all_us, ['forward', 'inverse'])
    utils.timer.toc('ode_auto_linear3')

  if 'ot_rho0param' in FLAGS.eqn_types:
    '''
    parameters: initial density rho_0
    condition/qoi: terminal rho_1 and rho_t
    this is similar to mfc_rhoparam_hj during training
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
    init_rhos = datagen.generate_gaussian_process(next(rng), us, eqns, kernel = datagen.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (eqns, nu)
    init_rhos = jax.nn.softplus(init_rhos) # (eqns, nu), positive
    init_rhos = init_rhos / jnp.mean(init_rhos, axis = -1, keepdims = True) # (eqns, nu)
    all_rhos_key = []; all_rhos_value = []; all_rho1s_key = []; all_rho1s_value = []; all_params = []
    for i, init_rho in enumerate(init_rhos):
      for j in range(quests):
        g_batch = datagen.generate_gaussian_process(next(rng), us, num, kernel = datagen.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
        g_batch = g_batch - jnp.mean(g_batch, axis = -1, keepdims = True) # (num, nu)
        init_rho_batch = einshape("i->ji", init_rho, j = num) # (num, nu)
        rhos, _ = mfc_hj.solve_mfc_periodic_batch(g_batch/run_cost, us, init_rho_batch, us, xs_grids, ts_grids, terminal_t, diffusion_eps, half_unroll_nums)
        all_rhos_key.append(np.array(txs_grids_batch)) # (num, (nt+1) * nx, 2)
        all_rhos_value.append(np.array(rhos[...,None])) # (num, (nt+1) * nx, 1)
        rho1s = rhos[:,-N:, None] # (num, nx, 1)
        all_rho1s_key.append(np.array(einshape("i->jik", xs, j = num, k = 1))) # (num, nx, 1)
        all_rho1s_value.append(np.array(rho1s)) # (num, nx, 1)
        all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(init_rho[0], init_rho[nu//3], init_rho[2*nu//3], init_rho[-1]))
      if i % 100 == 0:
        print(""); print(i, end = "", flush=True)
      print(".", end = "", flush = True)
  
    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_mfc_rhoparam_hj_tfrecord(name = name, eqn_type = "mfc_rhoparam_ot", all_params = all_params, 
                                all_rhos_key = all_rhos_key, all_rhos_value = all_rhos_value, 
                                all_gs_key = all_rho1s_key, all_gs_value = all_rho1s_value,
                                problem_types = ["forward11", "forward12"], nt = Nt+1, nx = N)


  if 'ot_rho1param' in FLAGS.eqn_types:
    '''
    parameters: terminal density rho_1
    condition/qoi: initial rho_0 and rho_t
    this is similar to mfc_gparam_hj during training
    with same conditions and qois
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
    terminal_rhos = datagen.generate_gaussian_process(next(rng), us, eqns, kernel = datagen.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (eqns, nu)
    terminal_rhos = jax.nn.softplus(terminal_rhos) # (eqns, nu), positive
    terminal_rhos = terminal_rhos / jnp.mean(terminal_rhos, axis = -1, keepdims = True) # (eqns, nu)
    all_rhos_key = []; all_rhos_value = []; all_rho0s_key = []; all_rho0s_value = []; all_params = []
    for i, terminal_rho in enumerate(terminal_rhos):
      for j in range(quests):
        g_batch = datagen.generate_gaussian_process(next(rng), us, num, kernel = datagen.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
        g_batch = g_batch - jnp.mean(g_batch, axis = -1, keepdims = True) # (num, nu)
        terminal_rho_batch = einshape("i->ji", terminal_rho, j = num) # (num, nu)
        rhos, _ = mfc_hj.solve_mfc_periodic_time_reversal_batch(g_batch/run_cost, us, terminal_rho_batch, us, xs_grids, ts_grids, terminal_t, diffusion_eps, half_unroll_nums)
        all_rhos_key.append(np.array(txs_grids_batch)) # (num, (nt+1) * nx, 2)
        all_rhos_value.append(np.array(rhos[...,None])) # (num, (nt+1) * nx, 1)
        rho0s = rhos[:,:N, None] # (num, nx, 1)
        all_rho0s_key.append(np.array(einshape("i->jik", xs, j = num, k = 1))) # (num, nx, 1)
        all_rho0s_value.append(np.array(rho0s)) # (num, nx, 1)
        all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(terminal_rho[0], terminal_rho[nu//3], terminal_rho[2*nu//3], terminal_rho[-1]))
      if i % 100 == 0:
        print(""); print(i, end = "", flush=True)
      print(".", end = "", flush = True)
  
    name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
    datawrite.write_mfc_gparam_hj_tfrecord(name = name, eqn_type = "mfc_gparam_ot", 
                                           all_params = all_params, all_txs = all_rhos_key, all_rhos = all_rhos_value,
                                          problem_types = ["forward11", "forward12", "forward22"], nt = Nt+1, nx = N)


if __name__ == "__main__":

  import tensorflow as tf
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  tf.config.set_visible_devices([], device_type='GPU')

  FLAGS = flags.FLAGS
  flags.DEFINE_integer('num', 6, 'number of demos + 1')
  flags.DEFINE_integer('quests', 5, 'number of questions in each operator')
  flags.DEFINE_integer('eqns', 100, 'number of operators in the equation')
  flags.DEFINE_integer('length', 50, 'length of trajectory and control')
  flags.DEFINE_float('dt', 0.02, 'time step in dynamics')
  flags.DEFINE_float('dx', 0.02, 'dx in pdes')
  flags.DEFINE_integer('nu_nx_ratio', 1, 'nu_nx_ratio in mfc_hj')
  flags.DEFINE_string('name', 'test', 'name of the dataset')
  flags.DEFINE_string('dir', '.', 'name of the dataset')
  flags.DEFINE_list('eqn_types', [], 'list of equations for data generation')
  flags.DEFINE_integer('ood_coeff1_grids', 40, 'ood grids in the 1st axis')
  flags.DEFINE_integer('ood_coeff2_grids', 31, 'ood grids in the 2nd axis')
  flags.DEFINE_float('pde_bdry_left_min', -1.0, 'pde left boundary min value')
  flags.DEFINE_float('pde_bdry_left_max', 1.0, 'pde left boundary max value')
  flags.DEFINE_float('pde_bdry_right_min', -1.0, 'pde right boundary min value')
  flags.DEFINE_float('pde_bdry_right_max', 1.0, 'pde right boundary max value')

  flags.DEFINE_integer('seed', 202, 'random seed')

  app.run(main)
