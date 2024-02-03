from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
import weno_scheme
import numpy as np
import weno_roll
import matplotlib.pyplot as plt
import tabulate

def test_reconstruct(N, u_fn, u_integral_fn, get_w, get_u_roll):
  dx = 1/N
  x = jnp.linspace(0, 1, N+1)
  x_left = x[:-1]
  x_right = x[1:]
  u_value = u_fn(x)[:,None] # [N+1, 1]
  u_average = 1/dx * (u_integral_fn(x_right) - u_integral_fn(x_left))[:,None]
  v_minus_rhalf_all, v_plus_lhalf_all = weno_scheme.weno_reconstruct(u_average, get_w, get_u_roll, u_value[0], u_value[-1])
  reconstruct_minus = v_minus_rhalf_all[:-1,:] # [N+1, 1]
  reconstruct_plus = v_plus_lhalf_all[1:,:] # [N+1, 1]
  return u_value, reconstruct_minus, reconstruct_plus



def test_rhs(N, f_fn, gradf_fn, u_fn, u_integral_fn, get_w, get_u_roll):
  dx = 1/N
  x = jnp.linspace(0, 1, N+1)
  x_left = x[:-1]
  x_right = x[1:]
  u_value = u_fn(x)[:,None] # [N+1, 1]
  u_average = 1/dx * (u_integral_fn(x_right) - u_integral_fn(x_left))[:,None]
  u_roll = get_u_roll(u_average, u_value[0], u_value[-1], weno_scheme.coeff.roll_list)
  w_r, w_r_t = get_w(u_roll) 
  alpha = weno_scheme.get_scalar_alpha(u_average, gradf_fn)
  rhs = weno_scheme.get_rhs(f_fn, alpha, u_roll, dx, w_r, w_r_t)

  gt_flux = lambda x: -f_fn(u_fn(x))
  gt_rhs = ((gt_flux(x_right) - gt_flux(x_left)) / dx)[:,None]

  return gt_rhs, rhs

if __name__ == "__main__":

  u_fn = lambda x: jnp.sin(2 * jnp.pi * x)
  u_integral_fn = lambda x: -1/(2 * jnp.pi) * jnp.cos(2 * jnp.pi * x)
  f_fn = lambda x: 0.5 * x * x
  gradf_fn = lambda x: x
  errors_minus = []
  errors_plus = []
  errors_rhs = []
  Ns = []
  for N in [2**i for i in range(4, 13)]:
    gt, reconstruct_minus, reconstruct_plus = test_reconstruct(N, u_fn, u_integral_fn, weno_scheme.get_w_classic, weno_roll.get_u_roll_periodic)
    gt_rhs, rhs = test_rhs(N, f_fn, gradf_fn, u_fn, u_integral_fn, weno_scheme.get_w_classic, weno_roll.get_u_roll_periodic)
    error_minus = jnp.abs(gt - reconstruct_minus)
    error_plus = jnp.abs(gt - reconstruct_plus)
    error_rhs = jnp.abs(gt_rhs - rhs)
    errors_minus.append(jnp.max(error_minus))
    errors_plus.append(jnp.max(error_plus))
    errors_rhs.append(jnp.max(error_rhs))
    Ns.append(N)

  print(tabulate.tabulate(zip(Ns, errors_minus, errors_plus, errors_rhs), headers = ["N", "error_minus", "error_plus", "error_rhs"]))

  plt.loglog(Ns, errors_minus, 'ro--', label = "v-")
  plt.loglog(Ns, errors_plus, 'b^--', label = "v+")
  plt.loglog(Ns, errors_rhs, 'g^--', label = "rhs")
  plt.loglog(Ns, 1/np.array(Ns)**5, 'k--', label = "1/N^5")
  plt.legend()
  plt.xlabel("N (dx = 1/N)")
  plt.ylabel("L infinity error")
  plt.savefig("weno_reconstruct.png")

      
  