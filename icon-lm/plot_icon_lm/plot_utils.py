import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


label_map = {
    "mfc_gparam_hj_forward11": {'legend': "MFC $g$-parameter 1D -> 1D", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
    "mfc_gparam_hj_forward12": {'legend': "MFC $g$-parameter 1D -> 2D", 'linestyle': '--', 'marker': 'o', 'xlabel': r"$x$"},
    "mfc_gparam_hj_forward22": {'legend': "MFC $g$-parameter 2D -> 2D", 'linestyle': ':', 'marker': 'o', 'xlabel': r"$x$"},
    "mfc_rhoparam_hj_forward11": {'legend': r"MFC $\rho_0$-parameter 1D -> 1D", 'linestyle': '-', 'marker': 's', 'xlabel': r"$x$"},
    "mfc_rhoparam_hj_forward12": {'legend': r"MFC $\rho_0$-parameter 1D -> 2D", 'linestyle': ':', 'marker': 's', 'xlabel': r"$x$"},
    "mfc_gparam_solve_forward": {'legend': "MFC 1", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
    "mfc_rhoparam_solve_forward": {'legend': "MFC 2", 'linestyle': '--', 'marker': 'o', 'xlabel': r"$x$"},
    "ode_auto_const_forward": {'legend': "Forward problem of ODE 1", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$t$"},
    "ode_auto_const_inverse": {'legend': "Inverse problem of ODE 1", 'linestyle': ':', 'marker': 'o', 'xlabel': r"$t$"},
    "ode_auto_linear1_forward": {'legend': "Forward problem of ODE 2", 'linestyle': '-', 'marker': 's', 'xlabel': r"$t$"},
    "ode_auto_linear1_inverse": {'legend': "Inverse problem of ODE 2", 'linestyle': ':', 'marker': 's', 'xlabel': r"$t$"},
    "ode_auto_linear2_forward": {'legend': "Forward problem of ODE 3", 'linestyle': '-', 'marker': 'd', 'xlabel': r"$t$"},
    "ode_auto_linear2_inverse": {'legend': "Inverse problem of ODE 3", 'linestyle': ':', 'marker': 'd', 'xlabel': r"$t$"},
    "pde_poisson_spatial_forward": {'legend': "Forward Poisson equation", 'linestyle': '-', 'marker': 'o', 'xlabel': r"$x$"},
    "pde_poisson_spatial_inverse": {'legend': "Inverse Poisson equation", 'linestyle': ':', 'marker': 'o', 'xlabel': r"$x$"},
    "pde_porous_spatial_forward" : {'legend': "Forward linear reaction-diffusion", 'linestyle': '-', 'marker': 's', 'xlabel': r"$x$"},
    "pde_porous_spatial_inverse": {'legend': "Inverse linear reaction-diffusion", 'linestyle': ':', 'marker': 's', 'xlabel': r"$x$"},
    "pde_cubic_spatial_forward":  {'legend': "Forward nonlinear reaction-diffusion", 'linestyle': '-', 'marker': 'd', 'xlabel': r"$x$"},
    "pde_cubic_spatial_inverse": {'legend': "Inverse nonlinear reaction-diffusion", 'linestyle': ':', 'marker': 'd', 'xlabel': r"$x$"},
    "series_damped_oscillator": {'legend': "time series prediction", 'linestyle': '-', 'marker': '*', 'xlabel': r"$t$"},
    "series_damped_oscillator_forward": {'legend': "Forward damped oscillator", 'linestyle': '-', 'marker': '*', 'xlabel': r"$t$"},
    "series_damped_oscillator_inverse": {'legend': "Inverse damped oscillator", 'linestyle': ':', 'marker': '*', 'xlabel': r"$t$"},
}



def calculate_error(pred, label, mask):
    '''
    pred: [..., len, 1]
    label: [..., len, 1]
    mask: [..., len]
    '''
    mask = mask.astype(bool)
    error = np.linalg.norm(pred - label, axis = -1) # [..., len]
    error = np.mean(error, where = mask) 
    gt_norm_mean = np.mean(np.linalg.norm(label, axis = -1), where = mask)
    relative_error = error/gt_norm_mean
    return error, relative_error

def pattern_match(patterns, name):
    for pattern in patterns:
        if pattern in name:
            return True
    return False

def get_error_from_dict(result_dict, key, demo_num, caption_id):
    error, relative_error = calculate_error(result_dict[(key, 'pred', demo_num, caption_id)], 
                                            result_dict[(key, 'ground_truth')], 
                                            result_dict[(key, 'mask')])
    return error, relative_error



@jax.jit
def laplace_u(u, dx):
  uxx = (u[:-2] + u[2:] - 2*u[1:-1])/dx**2 
  uxx_left = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3])/dx**2
  uxx_right = (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4])/dx**2
  uxx = jnp.pad(uxx, (1, 1), mode='constant', constant_values = (uxx_left, uxx_right))
  return uxx
laplace_u_batch = jax.jit(jax.vmap(laplace_u, in_axes=(0, None)))
