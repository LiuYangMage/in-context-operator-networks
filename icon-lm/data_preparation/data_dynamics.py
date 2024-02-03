import jax
import jax.numpy as jnp
from collections import namedtuple
from functools import partial
import data_utils
from einshape import jax_einshape as einshape


'''
the semantics of jax.lax.scan() are given roughly by:

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
'''

# traj[0] = init, final is affected by control[-1]

def rk4_step(u, c, dt, rhs):
  k1 = dt * rhs(c, u)
  k2 = dt * rhs(c, u + 0.5 * k1)
  k3 = dt * rhs(c, u + 0.5 * k2)
  k4 = dt * rhs(c, u + k3)
  u_next = u + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
  return u_next, u

def euler_step(u, c, dt, rhs):
  u_next = u + dt * rhs(c, u)
  return u_next, u


@partial(jax.jit, static_argnums=(-1,))
def ode_auto_const_fn(init, control, dt, coeff_a, coeff_b, step_fn):
  rhs = lambda c, u: coeff_a * c + coeff_b
  f = partial(step_fn, rhs = rhs, dt = dt)
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

@partial(jax.jit, static_argnums=(-1,))
def ode_auto_linear1_fn(init, control, dt, coeff_a, coeff_b, step_fn):
  rhs = lambda c, u: (coeff_a * c * u + coeff_b)
  f = partial(step_fn, rhs = rhs, dt = dt)
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

@partial(jax.jit, static_argnums=(-1,))
def ode_auto_linear2_fn(init, control, dt, coeff_a1, coeff_a2, coeff_a3, step_fn):
  rhs = lambda c, u: coeff_a1 * u + coeff_a2 * c + coeff_a3
  f = partial(step_fn, rhs = rhs, dt = dt)
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

@partial(jax.jit, static_argnums=(-1,))
def ode_auto_linear3_fn(init, control, dt, coeff_a1, coeff_a2, coeff_a3, step_fn):
  rhs = lambda c, u: coeff_a1 * c * u + coeff_a2 * u + coeff_a3
  f = partial(step_fn, rhs = rhs, dt = dt)
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

ode_auto_const_batch_fn = jax.jit(jax.vmap(ode_auto_const_fn, [0,0, None, None, None, None], (0,0)), static_argnums=(-1,))
ode_auto_linear1_batch_fn = jax.jit(jax.vmap(ode_auto_linear1_fn, [0,0, None, None, None, None],(0,0)), static_argnums=(-1,))
ode_auto_linear2_batch_fn = jax.jit(jax.vmap(ode_auto_linear2_fn, [0,0, None, None, None, None, None],(0,0)), static_argnums=(-1,))
ode_auto_linear3_batch_fn = jax.jit(jax.vmap(ode_auto_linear3_fn, [0,0, None, None, None, None, None],(0,0)), static_argnums=(-1,))


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
    control = data_utils.generate_gaussian_process(subkey1, ts, num, kernel = data_utils.rbf_kernel_jax, k_sigma = k_sigma, k_l = k_l)
  init = jax.random.uniform(subkey2, (num,), minval = init_range[0], maxval = init_range[1])
  # traj[0] = init, final is affected by control[-1]
  _, traj = ode_batch_fn(init, control, dt, *coeffs, euler_step)
  ts_expand = einshape("i->ji", ts, j = num)
  return ts_expand[...,None], control[...,None], traj[...,None]


if __name__ == "__main__":
  from jax.config import config
  config.update('jax_enable_x64', True)

  # test du/dt = u, with ground truth u = exp(t)
  init = 1
  dt = 0.02
  ts = jnp.arange(50) * dt
  control = ts
  final, traj = ode_auto_linear2_fn(init, control, dt, 1.0, 0, 0, rk4_step)
  assert jnp.allclose(final, jnp.exp(ts[-1]+dt))
  assert jnp.allclose(traj, jnp.exp(ts))
