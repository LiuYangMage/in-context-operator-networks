from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from weno_solver import generate_weno_scalar_sol, generate_weno_euler_sol
import utils

def test_weno_burgers(b):
  xs = jnp.linspace(0.0, 1.0, 100, endpoint=False)
  dx = 0.01
  steps = 2000
  plot_stride = 200
  dt = 0.0005
  fn = jax.jit(lambda u: b * u * u)
  grad_fn = jax.jit(lambda u: 2 * b * u)
  init = jnp.concatenate([jnp.zeros((25,)), jnp.ones((50,)), jnp.zeros((25,))])[None,:,None] # (batch, N, 1)
  sol = generate_weno_scalar_sol(dx = dx, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn)
  for i in range(1):
    plt.figure(figsize= (18,7))
    for j in range(0,steps+1,plot_stride):
      plt.subplot(2,6,j//plot_stride+1)
      # blue line with red markers of size 2
      plt.plot(xs, sol[i, j,:,0], 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('t = {:.3f}\n integral = {:.5f}'.format(j * dt, dx * np.sum(sol[i, j,:,0])))
      plt.xlim([0,1])
      plt.ylim([np.min(sol[i,0,:,0])-0.1,np.max(sol[i,0,:,0])+0.1])
    plt.suptitle('du/dt + d({:.2f} u^2)/dx = 0'.format(b))
    plt.tight_layout()
    plt.savefig('burgers_b{:.2f}.png'.format(b), dpi = 600)
    plt.close('all')

def test_weno_euler(gamma, tube = "Sod", N = 100, batch_size = 1, tic_toc = False, plot_all = False, plot_terminal = True):
  # build Sod shock tube

  dx = 1.0 / N
  dt = 0.0001
  
  if tube == "Sod":
    rhos = jnp.array([1.0, 0.125])
    us = jnp.array([0.0, 0.0])
    ps = jnp.array([1.0, 0.1])
    steps = 2000
  elif tube == "Lax":
    rhos = jnp.array([0.445, 0.5])
    us = jnp.array([0.698, 0.0])
    ps = jnp.array([3.528, 0.571])
    steps = 1300

  init_rho = jnp.concatenate([jnp.ones((N//2,)) * rhos[0], jnp.ones((N//2,)) * rhos[1]])[None,:,None] # (batch, N, 1)
  init_u = jnp.concatenate([jnp.ones((N//2,)) * us[0], jnp.ones((N//2,)) * us[1]])[None,:,None] # (batch, N, 1)
  init_p = jnp.concatenate([jnp.ones((N//2,)) * ps[0], jnp.ones((N//2,)) * ps[1]])[None,:,None] # (batch, N, 1)
  init_m = init_rho * init_u # (batch, N, 1)
  init_E = init_p / (gamma - 1) + 0.5 * init_rho * init_u * init_u # (batch, N, 1)
  init_U = jnp.concatenate([init_rho, init_m, init_E], axis = -1) # (batch, N, 3)
  init_U = jnp.tile(init_U, (batch_size,1,1)) # repeat batch_size times for time cost
  if tic_toc:
    out = utils.timeit(generate_weno_euler_sol)(dx, dt, gamma, init_U, steps, stable_tol = 1000) # (batch, steps + 1, N, 3)
    return out
  else:
    out = generate_weno_euler_sol(dx, dt, gamma, init_U, steps, stable_tol = 1000) # (batch, steps + 1, N, 3)
  
  if plot_all:
    # plot all solutions
    plt.figure(figsize= (30,7))
    for j in range(0,steps+1,100):
      # plot rho, v, and p, in three raws
      rho = out[0, j,:,0]
      u = out[0, j,:,1] / rho
      p = (gamma - 1) * (out[0, j,:,2] - 0.5 * rho * u * u)
      plt.subplot(3, steps//100 + 1, j//100+1)
      plt.plot(jnp.linspace(0.0, 1.0, N, endpoint=False), rho, 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('rho, t = {:.3f}'.format(j * dt))
      plt.xlim([0,1])
      # plt.ylim([0,1.1])
      plt.subplot(3, steps//100 + 1, j//100+1 + steps//100 + 1)
      plt.plot(jnp.linspace(0.0, 1.0, N, endpoint=False), u, 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('v, t = {:.3f}'.format(j * dt))
      plt.xlim([0,1])
      # plt.ylim([-0.1,1.1])
      plt.subplot(3, steps//100 + 1, j//100+1 + 2 * steps//100 + 2)
      plt.plot(jnp.linspace(0.0, 1.0, N, endpoint=False), p, 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('p, t = {:.3f}'.format(j * dt))
      plt.xlim([0,1])
      # plt.ylim([-0.1,1.1])

    plt.suptitle('Euler equations with gamma = {:.2f}'.format(gamma))
    plt.tight_layout()
    plt.savefig('euler_gamma_{}_{:.2f}.png'.format(tube,gamma), dpi = 600)
  
  if plot_terminal:
  # plot terminal solution
    j = steps
    rho = out[0, j,:,0]
    u = out[0, j,:,1] / rho
    p = (gamma - 1) * (out[0, j,:,2] - 0.5 * rho * u * u)

    plt.figure(figsize= (18,7))
    plt.subplot(1,3,1)
    plt.plot(jnp.linspace(0.0, 1.0, N, endpoint=False), rho, 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
    plt.title('rho, t = {:.3f}'.format(j * dt))
    plt.xlim([0,1])
    # plt.ylim([0,1.1])
    plt.subplot(1,3,2)
    plt.plot(jnp.linspace(0.0, 1.0, N, endpoint=False), u, 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
    plt.title('v, t = {:.3f}'.format(j * dt))
    plt.xlim([0,1])
    # plt.ylim([-0.1,1.1])
    plt.subplot(1,3,3)
    plt.plot(jnp.linspace(0.0, 1.0, N, endpoint=False), p, 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
    plt.title('p, t = {:.3f}'.format(j * dt))
    plt.xlim([0,1])
    # plt.ylim([-0.1,1.1])
    plt.suptitle('Euler equations with gamma = {:.2f}'.format(gamma))
    plt.tight_layout()
    plt.savefig('euler_gamma_{}_{:.2f}_terminal_N{}.png'.format(tube,gamma,N), dpi = 600)

  # close all figures
  plt.close('all')


if __name__ == "__main__":
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  # test_weno_burgers(0.5)
  # for batch_size in [1,10,100,500]:
  for batch_size in [1]:
    test_weno_euler(1.4, tube = "Sod", N = 160, batch_size = batch_size, tic_toc = False, plot_terminal = True)
    test_weno_euler(1.4, tube = "Lax", N = 160, batch_size = batch_size, tic_toc = False, plot_terminal = True)

    test_weno_euler(1.4, tube = "Sod", N = 400, batch_size = batch_size, tic_toc = False, plot_terminal = True)
    test_weno_euler(1.4, tube = "Lax", N = 400, batch_size = batch_size, tic_toc = False, plot_terminal = True)