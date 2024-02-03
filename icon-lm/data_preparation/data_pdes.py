from jax import lax
import jax.numpy as jnp
from functools import partial
from einshape import jax_einshape as einshape
from collections import namedtuple
import jax
jax.config.update("jax_enable_x64", True)

def tridiagonal_solve(dl, d, du, b): 
  """Pure JAX implementation of `tridiagonal_solve`.""" 
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1]) 
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_) 
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2]) 
  bwd1 = lambda x_, x: x[0] - x[1] * x_ 
  double = lambda f, args: (f(*args), f(*args)) 

  # Forward pass. 
  _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)), 
                    du[0] / d[0], 
                    (d, du, dl), 
                    unroll=32) 

  _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)), 
                  b[0] / d[0], 
                  (b, d, prepend_zero(tu_), dl), 
                  unroll=32) 

  # Backsubstitution. 
  _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)), 
                  b_[-1], 
                  (b_[::-1], tu_[::-1]), 
                  unroll=32) 

  return x_[::-1] 

@partial(jax.jit, static_argnames=("N"))
def solve_poisson(L, N, u_left, u_right, c):
    '''
    du/dxx = c over domain [0,L]
    c: spatially varying function, size N-1,
    u_left, u_right: boundary conditions. 
    the output is the full solution, (N+1) grid point values.  
    '''
    dx = L / N
    # x = jnp.linspace(0, L, N+1)

    # finite difference matrix
    du = jnp.array([1.0] * (N-2) + [0.0])
    dl =  jnp.array([0.0] + [1.0] * (N-2))
    d = - 2.0 * jnp.ones((N-1,))

    b = c*dx*dx
    b = b.at[0].add(-u_left)
    b = b.at[-1].add(-u_right)

    out_u = tridiagonal_solve(dl, d, du, b)
    u = jnp.pad(out_u, (1, 1), mode='constant', constant_values=(u_left, u_right))
    return u

@partial(jax.jit, static_argnames=("N"))
def solve_porous(L, N, u_left, u_right, a, k, c):
    '''
    - a u_xx + k(x) u = c, a > 0, k(x) > 0
    over domain [0,L]
    a, c : constants
    k(x) : spatially varying coefficient, size (N-1,), should be positive
    u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
    the output is the full solution, (N+1) grid point values.  
    '''
    dx = L / N
    # x = jnp.linspace(0, L, N+1)

    # finite difference matrix
    du = - a * jnp.array([1.0] * (N-2) + [0.0])
    dl = - a * jnp.array([0.0] + [1.0] * (N-2))
    d =  (a * 2.0) * jnp.ones((N-1,)) + k * dx * dx

    b = c*dx*dx*jnp.ones((N-1,))
    b = b.at[0].add(a * u_left)
    b = b.at[-1].add(a * u_right)

    out_u = tridiagonal_solve(dl, d, du, b)
    u = jnp.pad(out_u, (1, 1), mode='constant', constant_values=(u_left, u_right))
    return u

@jax.jit
def laplace_u(u, dx):
  uxx = (u[:-2] + u[2:] - 2*u[1:-1])/dx**2 
  uxx_left = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3])/dx**2
  uxx_right = (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4])/dx**2
  uxx = jnp.pad(uxx, (1, 1), mode='constant', constant_values = (uxx_left, uxx_right))
  return uxx
laplace_u_batch = jax.jit(jax.vmap(laplace_u, in_axes=(0, None)))

@partial(jax.jit, static_argnames=("N"))
def solve_square(L, N, u, u_left, u_right, a, k):
    '''
    - a u_xx + k u^2 = c(x), a > 0, k > 0
    over domain [0,L]
    u_left, u_right, a, k : constant parameters
    c(x) : spatially varying coefficient, size (N+1,)
    u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
    the output is the full solution and k, (N+1) grid point values.  
    u: a given profile (possibly a GP), need to be matched with u_left, u_right
      size [N+1,]
    '''
    dx = L / N
    new_u = u + jnp.linspace(u_left - u[0], u_right - u[-1], N+1)
    uxx = laplace_u(new_u, dx)
    c = -a * uxx+ k * new_u **2
    return new_u,c
    
@partial(jax.jit, static_argnames=("N"))
def solve_cubic(L, N, u, u_left, u_right, a, k):
    '''
    - a u_xx + k u^3 = c(x), a > 0, k > 0
    over domain [0,L]
    u_left, u_right, a, k : constant parameters
    c(x) : spatially varying coefficient, size (N+1,)
    u_left, u_right:  u(0)=b0, u(1)=b1, boundary conditions. 
    the output is the full solution and k, (N+1) grid point values.  
    u: a given profile (possibly a GP), need to be matched with u_left, u_right
      size [N+1,]
    '''
    dx = L / N
    new_u = u + jnp.linspace(u_left - u[0], u_right - u[-1], N+1)
    uxx = laplace_u(new_u, dx)
    c = -a * uxx+ k * new_u **3
    return new_u,c

solve_poisson_batch = jax.jit(jax.vmap(solve_poisson, in_axes=(None, None, None, None, 0)), static_argnums=(1,))
solve_porous_batch = jax.jit(jax.vmap(solve_porous, in_axes=(None, None, None, None, None, 0, None)), static_argnums=(1,))
solve_square_batch = jax.jit(jax.vmap(solve_square, in_axes=(None, None, 0, None, None, None, None)), static_argnums=(1,))
solve_cubic_batch = jax.jit(jax.vmap(solve_cubic, in_axes=(None, None, 0, None, None, None, None)), static_argnums=(1,))

if __name__ == "__main__":
    import numpy as np
    out = solve_poisson(L = 1, N = 100, u_left = 1, u_right = 1, c = np.ones((99,)))
    print(out.shape, out)

    k_spatial = np.zeros((99,))
    out_porous = solve_porous(L = 1, N = 100, u_left = 1, u_right = 1, a = -1.0, k = k_spatial, c = 1.0)
    assert np.allclose(out, out_porous)

    out_poc = solve_porous(L = 1, N = 100, u_left = 1, u_right = 1, a = 0, k = np.ones((99,)), c = 1.0)
    assert np.allclose(out_poc, np.ones((101,), dtype=np.float64))

    for i in range(5):
      a = np.random.uniform(0.5, 1)
      k = np.random.uniform(0.5, 1, size=(10,99))
      c = np.random.uniform(-1, 1)
      out_batch = solve_porous_batch(1, 100, 1, 1, a, k, c)
      for j in range(10):
        out = solve_porous(L = 1, N = 100, u_left = 1, u_right = 1, a = a, k = k[j], c = c)
        assert np.allclose(out, out_batch[j])
        res = -a * (out[:-2] + out[2:] - 2*out[1:-1])/(0.01)**2 + k[j] * out[1:-1] - c
        assert np.allclose(res, np.zeros((99,), dtype=np.float64))


    x = np.linspace(0, 1, 101)
    u_input = np.cos(x)
    [new_u,out_c] = solve_square(L = 1, N = 100, u = u_input, u_left = 1, u_right = 0.1, a = -1.0, k = 1.0)
    print(out_c.shape, out_c)
    assert np.allclose(new_u[0], 1.0)
    assert np.allclose(new_u[-1], 0.1)

    x = np.linspace(0, 1, 101)
    u_input = np.cos(x)
    [new_u,out_c] = solve_cubic(L = 1, N = 100, u = u_input, u_left = 1, u_right = 0.1, a = -1.0, k = 1.0)
    print(out_c.shape)
    assert np.allclose(new_u[0], 1.0)
    assert np.allclose(new_u[-1], 0.1)
    

