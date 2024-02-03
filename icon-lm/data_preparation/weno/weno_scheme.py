import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import weno_3_coeff as coeff


def get_euler_eigen_vector(u, c, H, gamma):
  # each of size [...]
  # right eigen vectors of Euler equations, R
  # [3, 3, ...]
  #      _                    _ 
  #     |                      |
  #     |   1      1       1   |
  #     |                      |
  # R = |  u-c     u      u+c  |
  #     |                      |
  #     |  H-uc   u^2/2   H+uc |
  #     |_                    _|
  v1 = jnp.stack([jnp.ones_like(u), jnp.ones_like(u), jnp.ones_like(u)], axis = 0) # [3, ...]
  v2 = jnp.stack([u - c,            u,                u + c           ], axis = 0) # [3, ...]
  v3 = jnp.stack([H - u * c,        0.5 * u**2,       H + u * c       ], axis = 0) # [3, ...]
  R = jnp.stack([v1, v2, v3], axis = 0) # [3, 3, ...]
  
  # left eigen vectors of Euler equations, R^{-1}
  # [3, 3, ...]
  #                          _                                       _ 
  #                         |                                         |
  #                         |  uc/(gamma-1)+u^2/2  -c/(gamma-1)-u   1 |
  #                         |                                         |
  # R^{-1}=(gamma-1)/(2c^2)*|  2(H-u^2)             2u             -2 |
  #                         |                                         |
  #                         | -uc/(gamma-1)+u^2/2   c/(gamma-1)-u   1 |
  #                         |_                                       _|
  l1 = jnp.stack([u * c / (gamma-1) + 0.5 * u**2,  -c / (gamma-1) - u,  jnp.ones_like(u)     ], axis = 0) # [3, ...]
  l2 = jnp.stack([2 * (H - u**2),                  2 * u,               -2 * jnp.ones_like(u)], axis = 0) # [3, ...]
  l3 = jnp.stack([-u * c / (gamma-1) + 0.5 * u**2, c / (gamma-1) - u,   jnp.ones_like(u)     ], axis = 0) # [3, ...]
  R_inv = (gamma-1)/(2*c**2) * jnp.stack([l1, l2, l3], axis = 0) # [3, 3, ...]

  return R, R_inv # [3, 3, ...]

@partial(jax.jit)
def get_euler_properties(U, gamma):
  # U: [..., 3], with or without batch
  rho, m, E = U[...,0], U[...,1], U[...,2]
  u = m / rho # [...]
  p = (gamma - 1) * (E - 0.5 * m * u)
  c = jnp.sqrt(gamma * p / rho)
  H = (E + p) / rho
  eigenvalues = jnp.array([u - c, u, u + c]) # [3, ...]
  return rho, u, p, c, H, eigenvalues

@partial(jax.jit, static_argnums=(1,2))
def get_scalar_alpha(u, grad_fn, gs = 100, redundancy = 0.1):
  umin, umax = jnp.min(u), jnp.max(u)
  u_range = jnp.linspace(umin, umax, gs)
  grad_fn_u = grad_fn(u_range)
  alpha = jnp.max(jnp.abs(grad_fn_u))
  return alpha * (1+redundancy)

get_scalar_alpha_batch = jax.jit(jax.vmap(get_scalar_alpha, in_axes = [0, None, None, None]), static_argnums=(1,2))

@partial(jax.jit, static_argnames=("boundary_type",))
def get_tv(u, left_bound, right_bound, boundary_type):
  if boundary_type == "dirichlet":
    left = jnp.concatenate([left_bound[None,:], u])
    right = jnp.concatenate([u, right_bound[None,:]])
    return jnp.sum(jnp.abs(left - right))
  elif boundary_type == "periodic":
    return jnp.sum(jnp.abs(u - jnp.roll(u, 1, axis = 0)))
  else:
    raise NotImplementedError

@partial(jax.jit, static_argnames=("f",))
def flux_Lax_Friedrichs(f, alpha, a, b):
  h = 0.5 * (f(a) + f(b) - alpha * (b-a))
  return h


@partial(jax.jit, static_argnames=("r",))
def get_v_r_rhalf(u_roll, r):
  '''
  u_roll: [usize + 2, udim, 5]
  v^{(r)}_{i+1/2}, LHS of 2.51
  '''
  v_r = jnp.einsum('jki,i->jk', u_roll[:,:,coeff.k-1-r:2*coeff.k-1-r], coeff.c_rj[r+1,:])
  return v_r

@partial(jax.jit, static_argnames=("r",))
def get_v_r_lhalf(u_roll, r):
  '''
  v^{(r)}_{i-1/2}
  '''
  v_r = jnp.einsum('jki,i->jk', u_roll[:,:,coeff.k-1-r:2*coeff.k-1-r], coeff.c_rj[r,:])
  return v_r

@jax.jit
def get_w_d(u_roll):
  '''
  u_roll: [usize + 2, udim, 5]
  '''
  usize, udim, _ = u_roll.shape
  w_r = jnp.tile(coeff.d_r, [usize,udim,1])
  w_r_t = jnp.tile(coeff.d_r_t, [usize,udim,1])
  return w_r, w_r_t


@jax.jit
def get_w_classic(u_roll):
  '''
  u_roll: [usize + 2, udim, 5]
  '''
  beta = coeff.get_beta(u_roll) # (usize + 2, udim, 3)
  alpha_r = coeff.d_r[None,None,:]/(coeff.epsilon + beta)**2 # (usize + 2, udim, 3)
  w_r = alpha_r / jnp.sum(alpha_r, axis = -1, keepdims=True) # (usize + 2, udim, 3)
  
  alpha_r_t = coeff.d_r_t[None,None,:]/(coeff.epsilon + beta)**2 # (usize + 2, udim, 3)
  w_r_t = alpha_r_t / jnp.sum(alpha_r_t, axis = -1, keepdims=True) # (usize + 2, udim, 3)
  return w_r, w_r_t



@jax.jit
def get_v_candidates(u_roll):
  '''
  get the candidates for weighted summation
  return [usize + 2, udim, 3], two ghost points
  '''
  v_minus_rhalf_c = jnp.stack([get_v_r_rhalf(u_roll,r) for r in range(coeff.k)], axis = -1) #candidates for v^-_{i+1/2}
  v_plus_lhalf_c = jnp.stack([get_v_r_lhalf(u_roll,r) for r in range(coeff.k)], axis = -1) #candidates for v^+_{i-1/2}

  return v_minus_rhalf_c, v_plus_lhalf_c


@jax.jit
def get_v(u_roll, w_r, w_r_t):
  '''
  w_r, w_r_t: [usize + 2, udim, 3]
  output: [usize + 2, udim]
  '''
  v_minus_rhalf_c, v_plus_lhalf_c = get_v_candidates(u_roll) #[usize + 2, udim, 3]
  v_minus_rhalf_all = jnp.einsum("ijk,ijk->ij", v_minus_rhalf_c, w_r) #v^-_{i+1/2}
  v_plus_lhalf_all = jnp.einsum("ijk,ijk->ij", v_plus_lhalf_c, w_r_t) #v^+_{i-1/2}
  return v_minus_rhalf_all, v_plus_lhalf_all #[usize + 2, udim]


@partial(jax.jit, static_argnames=("f",))
def get_flux(f, alpha, u_roll, w_r, w_r_t):
  v_minus_rhalf_all, v_plus_lhalf_all = get_v(u_roll, w_r, w_r_t) #[usize + 2, udim]

  v_minus_lhalf = v_minus_rhalf_all[:-2,:]
  v_minus_rhalf = v_minus_rhalf_all[1:-1,:]
  v_plus_lhalf = v_plus_lhalf_all[1:-1,:]
  v_plus_rhalf = v_plus_lhalf_all[2:,:]

  f_rhalf = flux_Lax_Friedrichs(f, alpha, v_minus_rhalf, v_plus_rhalf) #[usize, udim]
  f_lhalf = flux_Lax_Friedrichs(f, alpha, v_minus_lhalf, v_plus_lhalf)
  return f_rhalf, f_lhalf


@partial(jax.jit, static_argnames=("f",))
def get_rhs(f, alpha, u_roll, dx, w_r, w_r_t):
  f_rhalf, f_lhalf = get_flux(f, alpha, u_roll, w_r, w_r_t) #[usize, udim]
  rhs = - (f_rhalf - f_lhalf) / dx #[usize, udim]
  return rhs

@partial(jax.jit, static_argnames=("get_w", "get_u_roll"))
def weno_reconstruct(u, get_w, get_u_roll, left_bound, right_bound):
  this_u_roll = get_u_roll(u, left_bound, right_bound, coeff.roll_list) #[usize + 2, udim, 5]
  w_r, w_r_t = get_w(this_u_roll) #[usize + 2, udim, 3]
  v_minus_rhalf_all, v_plus_lhalf_all = get_v(this_u_roll, w_r, w_r_t)  #[usize, udim]
  return v_minus_rhalf_all, v_plus_lhalf_all


@partial(jax.jit, static_argnums=(3, 4, 5, 7))
def weno_step(dt, dx, u, get_w, get_u_roll, f, alpha, scheme, left_bound, right_bound):
  def get_current_rhs(this_u):
    this_u_roll = get_u_roll(this_u, left_bound, right_bound, coeff.roll_list) #[usize + 2, udim, 5]
    w_r, w_r_t = get_w(this_u_roll) #[usize + 2, udim, 3]
    rhs = get_rhs(f, alpha, this_u_roll, dx, w_r, w_r_t)  #[usize, udim]
    return rhs
  if scheme == "euler":
    rhs = get_current_rhs(u)
    new_u = u + dt * rhs
    return new_u
  elif scheme == "rk3":
    k1 = get_current_rhs(u)
    k2 = get_current_rhs(u + 0.5 * dt * k1)
    k3 = get_current_rhs(u - dt * k1 + 2 * dt * k2)
    new_u = u + dt/6*(k1 + 4 * k2 + k3)
    return new_u
  elif scheme == "rk4":
    k1 = get_current_rhs(u)
    k2 = get_current_rhs(u + 0.5 * dt * k1)
    k3 = get_current_rhs(u + 0.5 * dt * k2)
    k4 = get_current_rhs(u + dt * k3)
    new_u = u + dt/6*(k1 + 2 * k2 + 2 * k3 + k4) 
    return new_u
  else:
    raise NotImplementedError

weno_step_batch = jax.jit(jax.vmap(weno_step, in_axes = [None, None, 0, None, None, None, 0, None, 0, 0]), static_argnums=(3, 4, 5, 7))


def get_euler_eigen_on_interface(U_l, U_r, gamma, mode = "roe"):
  # ul, ur: [usize, udim]
  if mode == "simple":
    mean = 0.5 * (U_l + U_r)
    rho, u, p, c, H, eigenvalues = get_euler_properties(mean, gamma)
    R, R_inv = get_euler_eigen_vector(u, c, H, gamma)
    return eigenvalues, R, R_inv
  
  elif mode == "roe":
    rho_l, u_l, p_l, c_l, H_l, eigenvalues_l = get_euler_properties(U_l, gamma)
    rho_r, u_r, p_r, c_r, H_r, eigenvalues_r = get_euler_properties(U_r, gamma)
    # rho_mean = jnp.sqrt(rho_l * rho_r)
    u_mean = (jnp.sqrt(rho_l) * u_l + jnp.sqrt(rho_r) * u_r) / (jnp.sqrt(rho_l) + jnp.sqrt(rho_r))
    H_mean = (jnp.sqrt(rho_l) * H_l + jnp.sqrt(rho_r) * H_r) / (jnp.sqrt(rho_l) + jnp.sqrt(rho_r))
    c_mean = jnp.sqrt((gamma - 1) * (H_mean - 0.5 * u_mean**2))
    eigenvalues = jnp.array([u_mean - c_mean, u_mean, u_mean + c_mean]) # [3, usize]
    R, R_inv = get_euler_eigen_vector(u_mean, c_mean, H_mean, gamma)
    return eigenvalues, R, R_inv
  else:
    raise NotImplementedError


@partial(jax.jit, static_argnums=(3, 4, 6, 8))
def weno_step_euler(dt, dx, u, get_w, get_u_roll, gamma, f, alpha, scheme, left_bound, right_bound):
  average_mode = 'roe'
  def get_current_rhs(this_u):
    this_u_roll = get_u_roll(this_u, left_bound, right_bound, coeff.roll_list) #[usize + 2, udim, 5]
      
    right_ev, right_R, right_R_inv = get_euler_eigen_on_interface(this_u_roll[:,:,2], this_u_roll[:,:,3], gamma, average_mode) # [3, 3, usize + 2]
    char_roll = jnp.einsum("ijk,kjn->kin", right_R_inv, this_u_roll) #[usize + 2, udim, 5]
    w_r, w_r_t = get_w(char_roll) #[usize + 2, udim, 3]
    char_minus_rhalf_all, _ = get_v(char_roll, w_r, w_r_t) #[usize + 2, udim]
    v_minus_rhalf_all = jnp.einsum("ijk,kj->ki", right_R, char_minus_rhalf_all) #[usize + 2, udim]

    left_ev, left_R, left_R_inv = get_euler_eigen_on_interface(this_u_roll[:,:,1], this_u_roll[:,:,2], gamma, average_mode) # [3, 3, usize + 2]
    char_roll = jnp.einsum("ijk,kjn->kin", left_R_inv, this_u_roll) #[usize + 2, udim, 5]
    w_r, w_r_t = get_w(char_roll) #[usize + 2, udim, 3]
    _, char_plus_lhalf_all = get_v(char_roll, w_r, w_r_t) #[usize + 2, udim]
    v_plus_lhalf_all = jnp.einsum("ijk,kj->ki", left_R, char_plus_lhalf_all) #[usize + 2, udim]

    v_minus_lhalf = v_minus_rhalf_all[:-2,:] # [usize, udim]
    v_minus_rhalf = v_minus_rhalf_all[1:-1,:]
    v_plus_lhalf = v_plus_lhalf_all[1:-1,:]
    v_plus_rhalf = v_plus_lhalf_all[2:,:]

    char_minus_lhalf = char_minus_rhalf_all[:-2,:] # [usize, udim]
    char_minus_rhalf = char_minus_rhalf_all[1:-1,:]
    char_plus_lhalf = char_plus_lhalf_all[1:-1,:]
    char_plus_rhalf = char_plus_lhalf_all[2:,:]

    # du = R \alpha (v^+ - v^-)
    ev_abs_max = jnp.max(jnp.abs(right_ev[:, :-1]), axis = -1) # [3]

    du_rhalf =  jnp.einsum("ijk,kj->ki", right_R[:,:,1:-1], ev_abs_max * (char_plus_rhalf-char_minus_rhalf))  
    f_rhalf = 0.5 * (f(v_minus_rhalf) + f(v_plus_rhalf) - du_rhalf) #[usize, udim]

    du_lhalf =  jnp.einsum("ijk,kj->ki", left_R[:,:,1:-1], ev_abs_max * (char_plus_lhalf-char_minus_lhalf))
    f_lhalf = 0.5 * (f(v_minus_lhalf) + f(v_plus_lhalf) - du_lhalf)
    
    rhs = - (f_rhalf - f_lhalf) / dx #[usize, udim]
    return rhs
  
  if scheme == "euler":
    rhs = get_current_rhs(u)
    new_u = u + dt * rhs
    return new_u
  elif scheme == "rk3":
    k1 = get_current_rhs(u)
    k2 = get_current_rhs(u + 0.5 * dt * k1)
    k3 = get_current_rhs(u - dt * k1 + 2 * dt * k2)
    new_u = u + dt/6*(k1 + 4 * k2 + k3)
    return new_u
  elif scheme == "rk4":
    k1 = get_current_rhs(u)
    k2 = get_current_rhs(u + 0.5 * dt * k1)
    k3 = get_current_rhs(u + 0.5 * dt * k2)
    k4 = get_current_rhs(u + dt * k3)
    new_u = u + dt/6*(k1 + 2 * k2 + 2 * k3 + k4) 
    return new_u
  else:
    raise NotImplementedError
  
weno_step_euler_batch = jax.jit(jax.vmap(weno_step_euler, in_axes = [None, None, 0, None, None, None, None, 0, None, 0, 0]), static_argnums=(3, 4, 6, 8))


if __name__ == "__main__":
  U = jnp.array([2.1,3.3,4.7])[None,:]
  gamma = 1.4
  rho, u, p, c, H, eigenvalues = get_euler_properties(U, gamma)
  R, R_inv = get_euler_eigen_vector(u, c, H, gamma)
  print(eigenvalues[:,0])
  print(R[:,:,0])
  print(R_inv[:,:,0])
  print(R[:,:,0] @ R_inv[:,:,0])

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

  # check Roe average
  U_l = jnp.array([2.1,3.3,4.7])
  U_r = jnp.array([2.2,3.4,4.8])
  Lambda, R, R_inv = get_euler_eigen_on_interface(U_l, U_r, gamma, mode = "roe")
  # diagonal matrix of [u_mean - c_mean, u_mean, u_mean + c_mean]
  Lambda = jnp.diag(Lambda) # [3, 3]
  Jac = R @ Lambda @ R_inv # should approximate Jacobian of f(U_mean)
  RHS = Jac @ (U_r - U_l) # should approximate f(U_r) - f(U_l)
  LHS = euler_eqn(U_r, gamma) - euler_eqn(U_l, gamma)
  print(LHS, RHS)
  assert jnp.allclose(LHS, RHS)