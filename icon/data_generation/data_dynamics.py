import jax
import jax.numpy as jnp
from collections import namedtuple

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

DynamicsFn = namedtuple("DynamicsFn", ["name", "dyn", "dyn_batch"])

def build_ode_auto(func, name, dt = 0.01):
  '''
  constant control:
  du/dt = func(c(t), u(t))
  '''
  f = lambda u, c: (u + dt * func(c, u), u)
  @jax.jit
  def dyn(init, control):
    final, traj = jax.lax.scan(f, init, control)
    return (final, traj)
  dyn_batch = jax.jit(jax.vmap(dyn, [0,0],0))
  return DynamicsFn(name = "ode_auto_{}".format(name), dyn=dyn, dyn_batch = dyn_batch)


def build_ode_constant(func, name, dt = 0.01):
  '''
  constant control:
  du/dt = func(control)
  '''
  f = lambda u, c: (u + dt * func(c), u)
  @jax.jit
  def dyn(init, control):
    final, traj = jax.lax.scan(f, init, control)
    return (final, traj)
  dyn_batch = jax.jit(jax.vmap(dyn, [0,0],0))
  return DynamicsFn(name = "ode_const_{}".format(name), dyn=dyn, dyn_batch = dyn_batch)


def build_ode_linear(func, name, dt = 0.01):
  '''
  constant control:
  du/dt = func(control) * u
  '''
  f = lambda u, c: (u + dt * func(c) * u, u)
  @jax.jit
  def dyn(init, control):
    final, traj = jax.lax.scan(f, init, control)
    return (final, traj)
  dyn_batch = jax.jit(jax.vmap(dyn, [0,0],0))
  return DynamicsFn(name = "ode_linear_{}".format(name), dyn=dyn, dyn_batch = dyn_batch)


@jax.jit
def ode_auto_const_fn(init, control, dt, coeff_a, coeff_b):
  rhs = lambda c, u: coeff_a * c + coeff_b
  f = lambda u, c: (u + dt * rhs(c, u), u)
  # traj[0] = init, final is affected by control[-1]
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

@jax.jit
def ode_auto_linear1_fn(init, control, dt, coeff_a, coeff_b):
  rhs = lambda c, u: (coeff_a * c * u + coeff_b)
  f = lambda u, c: (u + dt * rhs(c, u), u)
  # traj[0] = init, final is affected by control[-1]
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

@jax.jit
def ode_auto_linear2_fn(init, control, dt, coeff_a1, coeff_a2, coeff_a3):
  rhs = lambda c, u: coeff_a1 * u + coeff_a2 * c + coeff_a3
  f = lambda u, c: (u + dt * rhs(c, u), u)
  # traj[0] = init, final is affected by control[-1]
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

@jax.jit
def ode_auto_linear3_fn(init, control, dt, coeff_a1, coeff_a2, coeff_a3):
  rhs = lambda c, u: coeff_a1 * c * u + coeff_a2 * u + coeff_a3
  f = lambda u, c: (u + dt * rhs(c, u), u)
  # traj[0] = init, final is affected by control[-1]
  final, traj = jax.lax.scan(f, init, control)
  return final, traj

ode_auto_const_batch_fn = jax.jit(jax.vmap(ode_auto_const_fn, [0,0, None, None, None], (0,0)))
ode_auto_linear1_batch_fn = jax.jit(jax.vmap(ode_auto_linear1_fn, [0,0, None, None, None],(0,0)))
ode_auto_linear2_batch_fn = jax.jit(jax.vmap(ode_auto_linear2_fn, [0,0, None, None, None, None],(0,0)))
ode_auto_linear3_batch_fn = jax.jit(jax.vmap(ode_auto_linear3_fn, [0,0, None, None, None, None],(0,0)))


if __name__ == "__main__":
  from jax.config import config
  config.update('jax_enable_x64', True)

  dyn_fn = build_ode_constant(lambda c: c, name = "x1", dt = 0.02)
  final, traj = dyn_fn.dyn(1.0, jnp.ones((10,)))
  print(dyn_fn.name, final, traj)

  dyn_fn = build_ode_auto(lambda c, u: c, name = "x1", dt = 0.02)
  final, traj = dyn_fn.dyn(1.0, jnp.ones((10,)))
  print(dyn_fn.name, final, traj)

  dyn_fn = build_ode_linear(lambda c: c, name = "x1", dt = 0.02)
  final, traj = dyn_fn.dyn(1.0, jnp.ones((10,)))
  print(dyn_fn.name, final, traj)

  dyn_fn = build_ode_auto(lambda c, u: c * u, name = "x1", dt = 0.02)
  final, traj = dyn_fn.dyn(1.0, jnp.ones((10,)))
  print(dyn_fn.name, final, traj)