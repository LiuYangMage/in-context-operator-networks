from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from einshape import jax_einshape as einshape
from functools import partial

# Define the covariance function
def rbf_kernel(x1, x2, sigma, l):
    """
    Radial basis function kernel
    """
    sq_norm = cdist(x1 / l, x2 / l, metric='sqeuclidean')
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_kernel_jax(x1, x2, sigma, l):
    """
    Radial basis function kernel, only support 1D x1 and x2
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1-xx2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_sin_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (jnp.sin(jnp.pi*(xx1-xx2)))**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

def rbf_circle_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    xx1_1 = jnp.sin(xx1 * 2 * jnp.pi)
    xx1_2 = jnp.cos(xx1 * 2 * jnp.pi)
    xx2_1 = jnp.sin(xx2 * 2 * jnp.pi)
    xx2_2 = jnp.cos(xx2 * 2 * jnp.pi)
    sq_norm = (xx1_1-xx2_1)**2/(l**2) + (xx1_2-xx2_2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

@partial(jax.jit, static_argnames=('num','kernel'))
def generate_gaussian_process(key, ts, num, kernel, k_sigma, k_l):
  '''
  ts: 1D array (length,)
  out: Gaussian process samples, 2D array (num, length)
  '''
  length = len(ts)
  mean = jnp.zeros((num,length))
  # cov = rbf_kernel(ts[:, None], ts[:, None], sigma=k_sigma, l=k_l)
  cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
  cov = einshape('ii->nii', cov, n = num)
  out = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
  return out


