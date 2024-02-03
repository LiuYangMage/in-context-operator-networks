import jax
import jax.numpy as jnp
from functools import partial

'''
u: [usize, udim]
lb: [udim]
rb: [udim]
out: [usize+2, udim]
'''
roll_db_funs = {2: jax.jit(lambda u, lb, rb: jnp.concatenate([jnp.tile(lb, [3,1]), u[:-1,:]], axis = 0)),
                1: jax.jit(lambda u, lb, rb: jnp.concatenate([jnp.tile(lb, [2,1]), u], axis = 0)),
                0: jax.jit(lambda u, lb, rb: jnp.concatenate([lb[None,:], u, rb[None,:]], axis = 0)),
               -1: jax.jit(lambda u, lb, rb: jnp.concatenate([u, jnp.tile(rb, [2,1])], axis = 0)),
               -2: jax.jit(lambda u, lb, rb: jnp.concatenate([u[1:,:], jnp.tile(rb, [3,1])], axis = 0)),
                }

roll_pb_funs = {2: jax.jit(lambda u: jnp.concatenate([u[-3:,:], u[:-1,:]], axis = 0)),
                1: jax.jit(lambda u: jnp.concatenate([u[-2:,:], u], axis = 0)),
                0: jax.jit(lambda u: jnp.concatenate([u[-1:,:], u, u[:1,:]], axis = 0)),
               -1: jax.jit(lambda u: jnp.concatenate([u, u[:2,:]], axis = 0)),
               -2: jax.jit(lambda u: jnp.concatenate([u[1:,:], u[:3,:]], axis = 0)),
                }

@partial(jax.jit, static_argnames =("roll_list",))
def get_u_roll_dirichlet(u, left_bound, right_bound, roll_list):
  '''
  u: [usize, udim]
  u_roll: [usize + 2, udim, 5]
  '''
  u_roll = jnp.stack([roll_db_funs[j](u, left_bound, right_bound) for j in roll_list], axis = -1) # stencil for u_{i}
  return u_roll



@partial(jax.jit, static_argnames =("roll_list",))
def get_u_roll_periodic(u, left_bound, right_bound, roll_list):
  '''
  u: [usize, udim]
  u_roll: [usize + 2, udim, 5]
  boundaries are dummy
  '''
  u_roll = jnp.stack([roll_pb_funs[j](u) for j in roll_list], axis = -1) # stencil for u_{i}
  return u_roll

if __name__ == "__main__":
  u = jnp.array([jnp.arange(10),jnp.arange(10)*2]).T
  lb = jnp.array([-99,-100])
  rb = jnp.array([99,100])
  r1 = get_u_roll_dirichlet(u, lb, rb, (2,1,0,-1,-2))
  print(r1[:,0,:].T)
  print(r1[:,1,:].T)
  print('----------')
  r2 = get_u_roll_periodic(u, lb, rb, (2,1,0,-1,-2))
  print(r2[:,0,:].T)
  print(r2[:,1,:].T)
