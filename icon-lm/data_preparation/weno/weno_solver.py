import jax
import jax.numpy as jnp

import weno_scheme
import weno_roll
import matplotlib.pyplot as plt
import numpy as np
import utils


@jax.jit
def euler_eqn(u, gamma):
  rho = u[...,0]
  rhv = u[...,1]
  ene = u[...,2]
  
  v = rhv/rho
  p = (gamma-1)*(ene - 0.5 * rhv * v)
  
  f1 = rhv
  f2 = rhv * v + p
  f3 = v * (ene + p)
  f = jnp.stack([f1,f2,f3], axis = -1) # [..., 3]
  return f


def generate_weno_scalar_sol(dx, dt, init, fn, steps, grad_fn = None, stable_tol = None):
  '''
  init: (batch, N, 1)
  '''
  alpha = weno_scheme.get_scalar_alpha_batch(init, grad_fn, 100, 0.1) # (batch,)
  left_bound = jnp.zeros_like(init) # dummy
  right_bound = jnp.zeros_like(init) # dummy
  us = [init]
  for i in range(steps):
    us.append(weno_scheme.weno_step_batch(dt, dx, us[-1], weno_scheme.get_w_classic, weno_roll.get_u_roll_periodic, fn, 
                          alpha, 'rk4', left_bound, right_bound))
  out = jnp.stack(us, axis = 1) # (batch, steps + 1, N, 1)
  # check if the solution is stable
  if stable_tol and (jnp.any(jnp.isnan(us[-1])) or jnp.max(jnp.abs(us[-1])) > stable_tol):
    print("sol instable", flush=True)
    for init_i, this_init in enumerate(init):
      sol = generate_weno_scalar_sol(dx = dx, dt = dt, init = this_init[None,...], fn = fn, steps = steps, grad_fn = grad_fn, stable_tol=None)
      if jnp.any(jnp.isnan(sol)) or jnp.max(jnp.abs(sol)) > 10.0:
        print(init_i)
        print(this_init)
    raise ValueError("sol contains nan")
  return out


def generate_weno_euler_sol(dx, dt, gamma, init, steps, stable_tol = None):
  '''
  init: (batch, N, 3)
  gamma: scalar
  '''
  fn = jax.jit(lambda u: euler_eqn(u, gamma))
  left_bound = init[:,0,:] # (batch, 3)
  right_bound = init[:,-1,:] # (batch, 3)
  us = [init]
  for i in range(steps):
    alpha = jnp.zeros((init.shape[0],)) # (batch,) dummy
    next_u = weno_scheme.weno_step_euler_batch(dt, dx, us[-1], 
                                         weno_scheme.get_w_classic, weno_roll.get_u_roll_dirichlet, 
                                         gamma, fn,
                                         alpha, 'rk4', left_bound, right_bound)
    us.append(next_u)
    
  out = jnp.stack(us, axis = 1) # (batch, steps + 1, N, 1)
  # check if the solution is stable
  if stable_tol and (jnp.any(jnp.isnan(us[-1])) or jnp.max(jnp.abs(us[-1])) > stable_tol):
    print("sol instable", flush=True)
    raise ValueError("sol contains nan")
  return out
