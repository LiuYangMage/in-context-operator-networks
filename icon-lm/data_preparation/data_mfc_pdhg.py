from jax.config import config
import os
config.update('jax_enable_x64', True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import plot
from einshape import jax_einshape as einshape
import math
from collections import namedtuple
import utils
import matplotlib.pyplot as plt


def cubic_poly_solve_vec(b,c,d):
  # find one real root of a 3rd order polynomial
  # root_vec = roots([1, cubic_coff1, 0,cubic_coff2]);
  #  cc = max(root_vec(imag(root_vec) == 0));
  #--- is equivalent as the one below
  #  cc=cubic_poly_solve(cubic_coff1,0,cubic_coff2); 
  # input b c d each of them is size (n,)
  # return (n,)
    b3over3=(b/3.)*(b/3.)*(b/3.)
        
    p=c-b*(b/3.)
    q=d+2.*b3over3-b*(c/3.)

    # solution=0
    real3rdRoot1=-0.5    #%equals cos(2*M_PI/3);
    # im3rdRoot1=0.86602540378   #%equals sin(2*M_PI/3);
    angle = 2.0*math.pi/3.0
    im3rdRoot1= jnp.sin(angle)  #%equals sin(2*M_PI/3);
    real3rdRoot2=-0.5  #equals cos(4*M_PI/3)=real3rdRoot1;
    # im3rdRoot2=-0.86602540378 #equals sin(4*M_PI/3)=-im3rdRoot1;  
    im3rdRoot2 = - im3rdRoot1

    mask0 = (p == 0)
    solution_p0=-jnp.sign(q)*jnp.exp(jnp.log(jnp.abs(q))/3.0)
    
    discrim=(q/2.)*(q/2.)+(p/3.)*(p/3.)*(p/3.)

    s=jnp.sqrt(abs(discrim))
    theta=jnp.arctan2(s,-q/2.)      
    x=s*s+q*q/4.
    rc=jnp.exp(jnp.log(x)/6.)    
    thetac=theta/3.
    real=rc*jnp.cos(thetac)
    im=rc*jnp.sin(thetac)
        
    solution1=2.*real
    solution2=2.*(real*real3rdRoot1-im*im3rdRoot1)
    solution3=2.*(real*real3rdRoot2-im*im3rdRoot2)
    solution_d_neg= jnp.maximum(solution1,jnp.maximum(solution2,solution3))

    u3=-q/2.+s
    v3=-q/2.-s
        
    u=jnp.sign(u3)*jnp.exp(jnp.log(jnp.abs(u3))/3.)
    v=jnp.sign(v3)*jnp.exp(jnp.log(jnp.abs(v3))/3.)
        
    solution_d_pos=u+v
    solution_d0=jnp.maximum(3.*q/p, -3.*q/(2.*p))


    # mask_discrim_neg = (discrim< -5e-10)
    # mask_discrim_pos = (discrim> 5e-10)
    mask_discrim_neg = (discrim< 0)
    mask_discrim_pos = (discrim> 0)
    mask_discrim_0 = (jnp.logical_not(mask_discrim_pos))*(jnp.logical_not(mask_discrim_neg))

    # solution_p0 = jnp.nan_to_num(solution_p0, nan=0.0) 
    # solution_d_neg = jnp.nan_to_num(solution_d_neg, nan=0.0) 
    # solution_d_pos = jnp.nan_to_num(solution_d_pos, nan=0.0) 
    # solution_d0 = jnp.nan_to_num(solution_d0, nan=0.0) 

    solution = solution_p0*mask0 + (jnp.logical_not(mask0))*(mask_discrim_neg*solution_d_neg + mask_discrim_pos *solution_d_pos + mask_discrim_0* solution_d0)
          
    solution = solution-b/3
    return solution



def helper_map1(l,k1, Nt, N1): 
  #index map for rho
  k1 = k1%N1
  return (l*N1 + k1)

def  helper_map2(l,k1, Nt, N1): 
  #index map for m1
  k1 = k1%N1
  return (Nt*N1 + l*N1 + k1)

def  helper_map3( l, k1, Nt, N1): 
  #index map for m2
  k1 = k1%N1
  return (2*Nt*N1 + l*N1 + k1)


@partial(jax.jit, static_argnums=(0,1))
def generate_FD_matrix_rhs(Nt,N1,dt, rho0):
  # mesh grid in dimension time, space x1, (x2)
  '''
  return 
    vector (N1xNt,), the first N1 entries are rho0/dt, the rest are zeros.
  '''
  b = rho0/dt
  b = jnp.pad(b, (0,  N1*Nt-N1), mode='constant', constant_values=(0,0))
  return b

generate_FD_matrix_rhs_batch = jax.jit(jax.vmap(generate_FD_matrix_rhs, in_axes=(None, None, None, 0), out_axes=0), 
                                       static_argnums=(0,1))

# setup Finite Difference matrix
def generate_FD_matrix(Nt,N1,dt,dx,viscosity):
  '''
  mesh grid in dimension time, space x1, x2
  @return:
    D: (N1*Nt, 3*N1*Nt)
  '''
  n = N1*Nt
  n3 = 3*n

  D = np.zeros((n,n3))
  l = 0
  for k1 in range(N1):
    index = helper_map1(l,k1,Nt,N1)
    D[index, index] = (1.0/dt + 2.0 * viscosity/dx**2)
    index0 = helper_map1(l,k1-1,Nt,N1)
    D[index, index0] = (-viscosity/dx**2)
    index0 = helper_map1(l,k1+1,Nt,N1)
    D[index, index0] = (-viscosity/dx**2)
    # flux
    index2 = helper_map2(l,k1, Nt,N1)
    D[index, index2] = (1.0/dx)
    index2 = helper_map2(l,k1-1, Nt,N1)
    D[index, index2] = (-1.0/dx)
    index3 = helper_map3(l,k1+1, Nt,N1)
    D[index, index3] = (1.0/dx)
    index3 = helper_map3(l,k1, Nt,N1)
    D[index,index3] = (-1.0/dx)


  for l in range(1,Nt):
    for k1 in range(N1):
      index = helper_map1(l,k1, Nt,N1)
      D[index, index]=(1.0/dt + 2.0 * viscosity/dx**2)
      index1 = helper_map1(l-1,k1, Nt,N1)
      D[index,index1]=(-1.0/dt)
      index0 = helper_map1(l,k1-1,Nt,N1)
      D[index, index0]=(-viscosity/dx**2)
      index0 = helper_map1(l,k1+1,Nt,N1)
      D[index, index0]=(-viscosity/dx**2)
      # flux
      index2 = helper_map2(l,k1,Nt,N1)
      D[index,index2]=(1.0/dx)
      index2 = helper_map2(l,k1-1,Nt,N1)
      D[index,index2]=(-1.0/dx)
      index3 = helper_map3(l,k1+1, Nt,N1)
      D[index,index3]=(1.0/dx)
      index3 = helper_map3(l,k1, Nt,N1)
      D[index,index3]=(-1.0/dx)

  D = jnp.array(D)
  return D


def initial_density(N1,dx,center = 0.5):
  '''
  initial density on domain [0, dx*N1]
  @return
    rho0: (N1,)
  '''
  sigma= 0.1
  x1 = jnp.arange(N1)*dx
  rho0 = 1.0/(2*math.pi*sigma**2) * jnp.exp(-(x1-center)**2/sigma**2)
  return rho0


def final_cost_fn(N1, dx, phase = 0.0):
  '''
  final cost on domain [0, dx*N1]
  @return
    g: (N1,)
  '''
  x1 = jnp.arange(N1)*dx
  g = 1.0 + jnp.sin(2*math.pi*(x1+ phase))
  return g

def run_cost_fn(Nt,N1,dx):
  '''
  running cost on domain [0, dx*N1]x[0, T]
  @return
    f: (Nt*N1)
  ''' 
  return jnp.ones((Nt*N1,)) * 10

MFCConfig = namedtuple("Config", ["dt", "dx", "D", 
                               "U", "UT", "V", "VT", "s_inv"])

def get_config(Nt, N1,viscosity):
  dt = 1/Nt
  dx = 1/N1
  # linear constraint, set up Finite Difference matrix and its SVD
  D = generate_FD_matrix(Nt,N1,dt,dx,viscosity) # (N1*Nt, 3*N1*Nt)
  U, s, VT = jnp.linalg.svd(D, full_matrices=False)
  s_inv = jnp.array([1./sing if sing >=1e-6 else 0 for sing in s])
  mfc_config = MFCConfig(dt=dt, dx=dx,
                  D=D, U=U, UT=jnp.transpose(U), V=jnp.transpose(VT), VT=VT, 
                  s_inv=s_inv)
  return mfc_config

def project_divergence(config, z, rhs):
  '''
  Projection to the subspace D x= rhs.
  z: (3*Nt*N1,)
  rhs: (Nt*N1,)
  '''
  res = config.D@z - rhs
  temp = config.V @ (config.s_inv*(config.UT@res))
  Pz = z - temp
  return Pz


def prox_L(Nt, N1, config, w, cost_c, final_cost, gamma):
  '''
  w: (3* Nt * N1 * N2) first n is rho, then 2n is m
  cost_c: (Nt * N1)
  final_cost (N1,)
  '''
  n = N1 * Nt
  coeff1 = - w[0:n]+ 4.0*gamma*cost_c # (Nt*N1,)
  coeff2 = - 4*gamma*cost_c*w[0:n] + (4.0*gamma*gamma)*cost_c*cost_c # (Nt*N1,)
  coeff3 = - gamma*cost_c*(w[2*n:3*n]**2 + w[n:2*n]**2) - (4*gamma*gamma)*cost_c*cost_c*w[0:n] # (Nt*N1,)

  # modify t=T
  coeff1 = coeff1.at[n-N1:n].add(gamma/config.dt*final_cost)
  coeff2 = coeff2.at[n-N1:n].add(gamma/config.dt*final_cost * (4*gamma)*(cost_c[n-N1:n]))
  coeff3 = coeff3.at[n-N1:n].add(gamma/config.dt*final_cost * (4*gamma**2)*(cost_c[n-N1:n])**2)

  # use cubic root finder 
  rho = cubic_poly_solve_vec(coeff1,coeff2,coeff3)
  rho = jnp.maximum(rho, 0.0)
  # reconstruct flux term
  m1 = rho*w[1*n:2*n]/(rho + 2*gamma*cost_c)
  m2 = rho*w[2*n:3*n]/(rho + 2*gamma*cost_c)

  m1 = jnp.maximum(m1, 0.0)
  m2 = jnp.minimum(m2, 0.0)
  out = jnp.concatenate([rho, m1, m2], 0)

  return out


@partial(jax.jit, static_argnums=(0,1,))
def apply_T(Nt, N1, config, z, w, rhs, final_cost, run_cost, gamma):
  # d = [rho0; G(x,T)] is MFG boundary conditions of size 2*(N1*N2) x 1 
  # z = approximate solution to MFG of size 3x N1*N2*Nt
  # cost_c = cost vector to be learned of size N1*N2*Nt
  c = run_cost
  #  D-R Splitting 
  w = w + prox_L(Nt, N1, config, 2*z - w, c, final_cost, gamma) - z
  z = project_divergence(config, w, rhs)
  return z,w

# vmap over z, w, rhs, final_cost, run_cost
apply_T_batch = jax.jit(jax.vmap(apply_T, in_axes=(None, None, None, 0, 0, 0, 0, 0, None)), 
                        static_argnums=(0,1,))


def forward(Nt, N1, config, rho0, final_cost, run_cost, gamma, max_iters=100, tol=1e-10, verbose=False):
  '''
  rho0: initial density, (N1,)
  final_cost: (N1,)
  run_cost: (Nt*N1,)
  x0: initial guess, (3*Nt*N1,)
  '''
  rhs = generate_FD_matrix_rhs(Nt, N1, config.dt, rho0)
  x0 = jnp.pad(jnp.tile(rho0, (Nt,)), ((0,2*Nt*N1)), 'constant') # (3*Nt*N1,)
  x_prev = x0
  x = x0
  w = x0

  for iter in range(max_iters):
    #  D-R Splitting 
    x,w = apply_T(Nt, N1, config, x, w, rhs, final_cost, run_cost, gamma)
    if verbose and (iter%300==0):
      fixed_pt_res = jnp.max(abs(x_prev - x))
      # loss0 = loss_L(x, c, final_cost)
      loss0 = 0
      print('iter = %2d, fixed_pt_res = %5.2e, loss =%5.2e' % (iter+1, fixed_pt_res, loss0 )) #print loss and res. 
      if fixed_pt_res < tol:
        break
    x_prev = x

  return x


def forward_batch(Nt, N1, config, rho0_batch, final_cost_batch, run_cost_batch, gamma, max_iters=1000, check_freq = 100, tol=1e-10, verbose=False):
  '''
  rho0: initial density, (batch, N1,)
  final_cost: final cost, (batch, N1,)
  x0: initial guess, (batch, 3*Nt*N1,)
  '''
  rhs = generate_FD_matrix_rhs_batch(Nt, N1, config.dt, rho0_batch)
  x0_batch = jnp.pad(jnp.tile(rho0_batch, (1,Nt)), ((0,0),(0,2*Nt*N1)), 'constant') # (batch, 3*Nt*N1,)
  x_prev = x0_batch
  x = x0_batch
  w = x0_batch
  for iter in range(max_iters):
    #  D-R Splitting 
    x,w = apply_T_batch(Nt, N1, config, x, w, rhs, final_cost_batch, run_cost_batch, gamma)
    if (iter+1) % check_freq == 0:
      x_prev = x
    if (iter%check_freq==0):
      fixed_pt_res = jnp.max(jnp.abs(x_prev - x), axis = -1) # max over x
      if verbose:
        print('iter = %2d, fixed_pt_res, mean = %5.2e, max = %5.2e' % (iter+1, 
                                jnp.mean(fixed_pt_res), jnp.max(fixed_pt_res))) #print loss and res. 
      if jnp.max(fixed_pt_res) < tol: # max over batch
        break
  return x, fixed_pt_res


@partial(jax.jit, static_argnums=(0,1,))
def loss_L(Nt, N1, config, w, cost_c, final_cost):
  # w: (3 * Nt * N1,), first n is rho, then 2n is m
  # cost_c: (Nt * N1,)
  # final_cost: (N1,)
  n = N1 * Nt
  rho = w[0:n]
  rho_inv = 1./rho
  rho_inv = jnp.nan_to_num(rho_inv, nan=0.0, posinf=0.0) # set nan and inf to 0, avoid zero density
  m_norm_sq =  w[n:2*n]*w[n:2*n] + w[2*n:3*n]*w[2*n:3*n]
  run_cost = jnp.sum(rho_inv * m_norm_sq * cost_c) * config.dx * config.dt
  final_cost = jnp.sum(final_cost * (rho[n-N1:n])) * config.dx 
  return run_cost + final_cost
  

loss_L_batch = jax.jit(jax.vmap(loss_L, in_axes=(None, None, None, 0, None, None)), 
                        static_argnums=(0,1,))

def plot_density(z,dx,N1, number0, vmin = 0, vmax = 3):
  z = np.array(z)
  fig = plt.figure(figsize =(3, 3))
  x = np.arange(0, 1, dx)
  for l in number0:
    rho_slice = z[l*N1:(l+1)*N1]
    mass = (sum(rho_slice)*dx)
    plt.plot(x, rho_slice, label = "t ind =" + str(l) + " mass = " + str(mass))
  return fig


def plot_inital_and_cost_functional(rho0,g,dx):
  # Creating dataset
  # x = np.linspace(0, 1, N1)
  fig = plt.figure(figsize =(3, 3))
  x = np.arange(0, 1, dx)
  z_rho0 = rho0
  z_g0 = g
  plt.plot(x, z_rho0, label = "initial rho")
  plt.plot(x, z_g0, label = "g")
  return fig


if __name__ == "__main__":
  pass

  
