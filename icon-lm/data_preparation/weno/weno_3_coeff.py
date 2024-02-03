import jax.numpy as jnp

k = 3
c_rj = jnp.array([
                [11/6, -7/6, 1/3], 
                [1/3, 5/6, -1/6], 
                [-1/6, 5/6, 1/3], 
                [1/3, -7/6, 11/6], 
                ])

d_r = jnp.array([3/10, 3/5, 1/10])
d_r_t = jnp.array([1/10, 3/5, 3/10])

roll_list = (2,1,0,-1,-2)

def get_beta(u_roll):
  '''
  u_roll: [..., 5], [2,1,0,-1,-2]
  '''
  us = [u_roll[..., i] for i in [2, 3, 4, 0, 1]] #[0,-1,-2,2,1]
  beta_0 = 13/12 * (us[0] - 2 * us[1] + us[2])**2 + 1/4 * (3 * us[0] - 4 * us[1] + us[2])**2
  beta_1 = 13/12 * (us[-1] - 2 * us[0] + us[1])**2 + 1/4 * (us[-1] - us[1])**2
  beta_2 = 13/12 * (us[-2] - 2 * us[-1] + us[0])**2 + 1/4 * (us[-2] - 4 * us[-1] + 3 * us[0])**2
  return jnp.stack([beta_0, beta_1, beta_2], axis = -1) # [..., 3]


epsilon = 1E-6