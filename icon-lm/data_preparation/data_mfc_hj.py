from jax.config import config
import os
config.update('jax_enable_x64', True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import plot
from einshape import jax_einshape as einshape
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import haiku as hk
import gc
from utils import timer
import data_utils


# phi(x,t) = -eps log \int exp(-1/eps(g(u) + (x-u)^2/2/t)) / sqrt(2pi* t* eps)du
def viscous_HJ_solver_1d_Riemann(g_u, u_grids, x, t, diffusion_eps, if_u_grids_full_domain = True):
    '''
    @parameters:
        g_u: [nu], or [n_pts, nu], initial condition g at grid points wrt u
        u_grids: [nu], or [n_pts, nu], grid points of u
        x: [n_pts]
        t: [n_pts]
        diffusion_eps: scalar
        if_u_grids_full_domain: bool, whether u_grids cover the full domain or just the domain of g
    @return:
        phi: [n_pts]
    '''
    quad = (x[:,None] - u_grids)**2/2/t[:,None]  # [n_pts, nu]
    g_plus_quad = g_u + quad  # [n_pts, nu]
    fn_val = -g_plus_quad/diffusion_eps  # [n_pts, nu]
    fn_max = jnp.max(fn_val, axis=1, keepdims=True)  # [n_pts,1]
    phi = -diffusion_eps * (jnp.log(jnp.sum(jnp.exp(fn_val - fn_max), axis=1)) + fn_max[:,0])  # [n_pts]
    if if_u_grids_full_domain:
        # to avoid numerical errors, instead of dividing by sqrt{2pi*t*eps}, we divide by sum exp(-(x-u)^2/2t/eps)
        phi = phi + diffusion_eps * jnp.log(jnp.sum(jnp.exp(-quad/diffusion_eps), axis=1))
    else:
        du = u_grids[1] - u_grids[0]
        phi = phi + diffusion_eps * jnp.log(2*jnp.pi * t * diffusion_eps) /2 - diffusion_eps * jnp.log(du)  # [n_pts]
    # print('shape x, t, u_grids, g_u, phi = {}, {}, {}, {}, {}'.format(x.shape, t.shape, u_grids.shape, g_u.shape, phi.shape), flush=True)
    return phi

def viscous_HJ_solver_1d_Riemann_periodic(g_u, u_grids, x, t, diffusion_eps, half_unroll_num = 6):
    '''
    @parameters:
        g_u: [nu], initial condition g at grid points wrt u
        u_grids: [nu], grid points of u
        x: [n_pts]
        t: [n_pts]
        diffusion_eps: scalar
        unroll_num: int, number of times to unroll the periodic domain
    @return:
        phi: [n_pts]
    '''
    # unroll u_grids and g_u
    du = u_grids[1] - u_grids[0]
    nu = u_grids.shape[0]
    u_period = du * nu
    unroll_num = 2*half_unroll_num + 1
    g_u_unroll = jnp.concatenate([g_u for _ in range(unroll_num)], axis=0)  # [nu_repeated = nu*unroll_num]
    u_grids_unroll = jnp.concatenate([u_grids + i*u_period for i in range(-half_unroll_num, half_unroll_num+1)], axis=0)  # [nu_repeated]
    # rotate g_u such that centered at x
    # u_center = (u_grids[0] + u_grids[-1])/2
    # n_rotation = jnp.round((x - u_center)/du)  # [n_pts]
    # g_u_tile = einshape('i->ji', g_u_unroll, j=x.shape[0])  # [n_pts, nu_repeated]
    # roll_vmap = jax.vmap(jnp.roll, [0,0], 0)
    # g_u_rotated = roll_vmap(g_u_tile, -n_rotation)  # [n_pts, nu_repeated]
    # u_shifted = u_grids_unroll + n_rotation[:,None]*du  # [n_pts, nu_repeated]
    # phi = viscous_HJ_solver_1d_Riemann(g_u_rotated, u_shifted, x, t, diffusion_eps)
    phi = viscous_HJ_solver_1d_Riemann(g_u_unroll, u_grids_unroll, x, t, diffusion_eps)
    return phi

# u_PM = (\int u exp(-1/eps(g(u) + (x-u)^2/2/t))du)/ (\int exp(-1/eps(g(u) + (x-u)^2/2/t))du)
def viscous_prox_1d_Riemann(g_u, u_grids, x, t, diffusion_eps):
    '''
    @parameters:
        g_u: [nu], initial condition g at grid points wrt u
        u_grids: [nu], grid points of u
        x: [n_pts]
        t: [n_pts]
    @return:
        u_pm: [n_pts]
    '''
    g_plus_quad = g_u + (x[:,None] - u_grids)**2/2/t[:,None]  # [n_pts, nu]
    fn_val = -g_plus_quad/diffusion_eps  # [n_pts, nu]
    fn_max = jnp.max(fn_val, axis=1, keepdims=True)  # [n_pts,1]
    denominator = jnp.sum(jnp.exp(fn_val - fn_max), axis=1)  # [n_pts]
    numerator = jnp.sum(jnp.exp(fn_val - fn_max) * u_grids, axis=1)  # [n_pts]
    u_pm = numerator / denominator
    # print('shape x, t, u_grids, g_u, numerator, denominator = {}, {}, {}, {}, {}, {}'.format(x.shape, t.shape, u_grids.shape, 
    #                                                 g_u.shape, numerator.shape, denominator.shape), flush=True)
    return u_pm

# grad_x phi(x,t) = (x - u_pm)/t
def viscous_HJ_gradx_1d_Riemann(g_u, u_grids, x, t, diffusion_eps):
    '''
    @parameters:
        g_u: [nu], initial condition g at grid points wrt u
        u_grids: [nu], grid points of u
        x: [n_pts]
        t: [n_pts]
    @return:
        gradx_phi: [n_pts]
    '''
    u_pm = viscous_prox_1d_Riemann(g_u, u_grids, x, t, diffusion_eps)
    gradx_phi = (x - u_pm)/t
    return gradx_phi

def extend_fn_periodic(fn_val, repeat_num = 1):
    '''
    extend a fn to a larger domain, by repeating fn vals (periodic fn) or padding a constant number
    @parameters:
        fn_val: [nx], function value at grid points
        repeat_num: int
    @return:
        fn_val_repeat: [nx * repeat_num], function value at grid points
    '''
    fn_val_repeat = einshape('j->(kj)', fn_val, k = repeat_num)  # [nx * repeat_num]
    return fn_val_repeat

def extend_fn_padding(fn_val, num_paddings, padding_val = 0.0):
    '''
    extend a fn to a larger domain, by padding a constant number
    @parameters:
        fn_val: [nx], function value at grid points
        num_padding: tuple of tuples of int, number of paddings at each side along each dim
    @return:
        fn_val_padded: [nx + num of padded zeros], function value at grid points
    '''
    fn_val_padded = jnp.pad(fn_val, num_paddings, 'constant', constant_values = padding_val)
    return fn_val_padded

def df_dt(f, dt):
    '''
    @parameters:
        f: [nx, nt]
        dt: float
    @return:
        df_dt: [nx, nt-1]
    '''
    return (f[:,1:] - f[:,:-1]) / dt

def df_dx(f, dx, periodic = False):
    '''
    @parameters:
        f: [nx, nt]
        dx: float
        periodic: bool
    @return:
        df_dx: [nx, nt] (if not periodic copy the same value at the boundary)
    '''
    if periodic:
        ret = (jnp.roll(f, -1, axis = 0) - jnp.roll(f, 1, axis = 0)) / (2*dx)
    else:
        ret = (f[2:,:] - f[:-2,:]) / (2*dx)
        ret = jnp.concatenate([ret[0:1,:], ret, ret[-1:,:]], axis = 0)
    return ret

def d2f_dx2(f, dx, periodic = False):
    '''
    @parameters:
        f: [nx, nt]
        dx: float
        periodic: bool
    @return:
        d2f_dx2: [nx, nt] (if not periodic pad zero at the boundary)
    '''
    if periodic:
        ret = (jnp.roll(f, -1, axis = 0) - 2*f + jnp.roll(f, 1, axis = 0)) / (dx**2)
    else:
        ret = (f[2:,:] - 2*f[1:-1,:] + f[:-2,:]) / (dx**2)
        ret = jnp.pad(ret, ((1,1),(0,0)), 'constant', constant_values = 0.0)
    return ret


# rho(x,t) = (\int mu(y,0)/rho(y,0) * exp(-(x-y)^2/2/eps/t) dy) / mu(x,t),
# where mu(x,t) is the solution of backward heat equation with initial condition 
@partial(jax.jit, static_argnames = ("if_u_grids_full_domain", "if_y_grids_full_domain"))
def solve_mfc_unbdd(g_u, u_grids, rho0_y, y_grids, x, t, terminal_time, diffusion_eps, 
                    if_u_grids_full_domain = True, if_y_grids_full_domain = True):
    '''
    @parameters:
        g_u: [nu], terminal cost g at grid points wrt u
        u_grids: [nu], grid points of u
        rho0_y: [ny], initial condition rho0 at grid points wrt y
        y_grids: [ny], grid points of y
        x: [n_pts]
        t: [n_pts]
        terminal_time: float
        diffusion_eps: float
        if_u_grids_full_domain: bool, if u_grids cover the full domain or just the domain of g
        if_y_grids_full_domain: bool, if y_grids cover the full domain or just the domain of rho0
    @return:
        rho: [n_pts]
        phi: [n_pts]
    '''
    # solve a backward viscous HJ using IC g
    terminal_time_arr = terminal_time + jnp.zeros_like(y_grids)  # [ny]
    phi_bkwd0 = viscous_HJ_solver_1d_Riemann(g_u, u_grids, y_grids, terminal_time_arr, diffusion_eps, if_u_grids_full_domain)  # [ny]
    # solve a forward viscous HJ using IC -eps log(rho0/exp(-phi_bkwd0/eps)) = -eps log(rho0) + phi_bkwd0
    phi_fwd0 = -diffusion_eps * jnp.log(rho0_y) - phi_bkwd0  # [ny]
    phi_fwdt = viscous_HJ_solver_1d_Riemann(phi_fwd0, y_grids, x, t, diffusion_eps, if_y_grids_full_domain)  # [n_pts]
    # rho = exp(-(phi_fwd + phi_bkwd)/eps)
    phi_bkwdt = viscous_HJ_solver_1d_Riemann(g_u, u_grids, x, terminal_time - t, diffusion_eps, if_u_grids_full_domain)  # [n_pts]
    g_interp = jnp.interp(x, u_grids, g_u)  # [n_pts]
    phi_bkwdt = jnp.where(t < terminal_time - 1e-6, phi_bkwdt, g_interp)  # [n_pts], if t is T, return interpolation of g
    rho = jnp.exp(-(phi_fwdt + phi_bkwdt)/diffusion_eps)  # [n_pts]
    rho0_interp = jnp.interp(x, y_grids, rho0_y)  # [n_pts]
    rho = jnp.where(t > 1e-6, rho, rho0_interp)  # [n_pts], if t is 0, return interp of rho0
    return rho, phi_bkwdt


def get_half_unroll_nums(u_grids, y_grids, t, terminal_time, diffusion_eps):
    '''
    get half unroll nums used in solve_mfc_periodic
    '''
    minimum = 2
    half_unroll_num1 = jnp.sqrt(20 * 2* diffusion_eps * terminal_time) // (u_grids[-1] - u_grids[0]) + 1
    half_unroll_num2 = jnp.sqrt(20 * 2* diffusion_eps * jnp.max(t)) // (y_grids[-1] - y_grids[0]) + 1
    half_unroll_num3 = jnp.sqrt(20 * 2* diffusion_eps * jnp.abs(terminal_time - jnp.min(t))) // (u_grids[-1] - u_grids[0]) + 1
    half_unroll_num1 = jnp.maximum(half_unroll_num1, minimum)
    half_unroll_num2 = jnp.maximum(half_unroll_num2, minimum)
    half_unroll_num3 = jnp.maximum(half_unroll_num3, minimum)
    return (int(half_unroll_num1), int(half_unroll_num2), int(half_unroll_num3))


# rho(x,t) = (\int mu(y,0)/rho(y,0) * exp(-(x-y)^2/2/eps/t) dy) / mu(x,t),
# where mu(x,t) is the solution of backward heat equation with initial condition 
@partial(jax.jit, static_argnums = (8,))
def solve_mfc_periodic(g_u, u_grids, rho0_y, y_grids, x, t, terminal_time, diffusion_eps, half_unroll_nums):
    '''
    @parameters:
        g_u: [nu], terminal cost g at grid points wrt u
        u_grids: [nu], grid points of u. [Note: Assume they come in order] 
        rho0_y: [ny], initial condition rho0 at grid points wrt y
        y_grids: [ny], grid points of y. [Note: Assume they come in order] 
        x: [n_pts]
        t: [n_pts]
        terminal_time: float
        diffusion_eps: float
        half_unroll_nums: (num1, num2, num3), interger tuple of size 3
    @return:
        rho: [n_pts]
        phi: [n_pts]
    '''
    period_g = (u_grids[1] - u_grids[0]) * len(u_grids)
    period_rho = (y_grids[1] - y_grids[0]) * len(y_grids)
    half_unroll_num1, half_unroll_num2, half_unroll_num3 = half_unroll_nums
    # solve a backward viscous HJ using IC g
    terminal_time_arr = terminal_time + jnp.zeros_like(y_grids)  # [ny]
    phi_bkwd0 = viscous_HJ_solver_1d_Riemann_periodic(g_u, u_grids, y_grids, terminal_time_arr, diffusion_eps, half_unroll_num1)  # [ny]
    # solve a forward viscous HJ using IC -eps log(rho0/exp(-phi_bkwd0/eps)) = -eps log(rho0) + phi_bkwd0
    phi_fwd0 = -diffusion_eps * jnp.log(rho0_y) - phi_bkwd0  # [ny]
    phi_fwdt = viscous_HJ_solver_1d_Riemann_periodic(phi_fwd0, y_grids, x, t, diffusion_eps, half_unroll_num2)  # [n_pts]
    # rho = exp(-(phi_fwd + phi_bkwd)/eps)
    phi_bkwdt = viscous_HJ_solver_1d_Riemann_periodic(g_u, u_grids, x, terminal_time - t, diffusion_eps, half_unroll_num3)  # [n_pts]
    g_interp = jnp.interp(x, u_grids, g_u, period = period_g)  # [n_pts], periodic interpolation
    phi_bkwdt = jnp.where(t < terminal_time - 1e-6, phi_bkwdt, g_interp)  # [n_pts], if t is T, return interpolation of g
    rho = jnp.exp(-(phi_fwdt + phi_bkwdt)/diffusion_eps)  # [n_pts]
    rho0_interp = jnp.interp(x, y_grids, rho0_y, period = period_rho)  # [n_pts], periodic interpolation
    rho = jnp.where(t > 1e-6, rho, rho0_interp)  # [n_pts], if t is close to 0, return interp of rho0
    return rho, phi_bkwdt

# solve mfc with initial HJ condition g and terminal rho1
@partial(jax.jit, static_argnums = (8,))
def solve_mfc_periodic_time_reversal(g_u, u_grids, rho1_y, y_grids, x, t, terminal_time, diffusion_eps, half_unroll_nums):
    time_reversed = terminal_time - t
    return solve_mfc_periodic(g_u, u_grids, rho1_y, y_grids, x, time_reversed, terminal_time, diffusion_eps, half_unroll_nums)

solve_mfc_periodic_batch = jax.jit(jax.vmap(solve_mfc_periodic, 
                                            in_axes = (0, None, 0, None, None, None, None, None, None), out_axes = (0,0)), 
                                            static_argnums = (8,))
solve_mfc_periodic_time_reversal_batch = jax.jit(jax.vmap(solve_mfc_periodic_time_reversal, 
                                            in_axes = (0, None, 0, None, None, None, None, None, None), out_axes = (0,0)), 
                                            static_argnums = (8,))


@partial(jax.jit, static_argnums=(9,))
def compute_mfc_residual_obj_periodic(gs, rho_0, us, xs, ts, terminal_t, diffusion_eps, dx_res, dt_res, half_unroll_nums):
    '''
    Note: gs and rho_0 use grids us. The solution is computed on grids (xs, ts).
        If t=0 or T, the error is not good because of linear interpolation and Euler forward fd for time derivatives
    @parameters:
        gs: [nu], terminal cost g at grid points wrt u
        rho_0: [nu], initial condition rho0 at grid points wrt u
        us: [nu], grid points of u
        xs: [n_pts], grid points of x
        ts: [n_pts], grid points of t
        terminal_t: float
        diffusion_eps: float
        dx_res: float
        dt_res: float
    @return:
        residual_cont, residual_HJ: (n_pts)
        obj_fn: sum of |gradx phi|^2* rho /2
    '''
    assert len(us) == len(gs) == len(rho_0)
    assert len(xs) == len(ts)
    period = (us[1] - us[0]) * len(us)
    xs_left = jnp.where(xs - dx_res >= us[0], xs - dx_res, xs - dx_res + period)
    xs_right = jnp.where(xs + dx_res <= us[0] + period, xs + dx_res, xs + dx_res - period)
    rho_middle, phi_middle = solve_mfc_periodic(gs, us, rho_0, us, xs, ts, terminal_t, diffusion_eps, half_unroll_nums)
    rho_left, phi_left = solve_mfc_periodic(gs, us, rho_0, us, xs_left, ts, terminal_t, diffusion_eps, half_unroll_nums)
    rho_right, phi_right = solve_mfc_periodic(gs, us, rho_0, us, xs_right, ts, terminal_t, diffusion_eps, half_unroll_nums)
    rho_up, phi_up = solve_mfc_periodic(gs, us, rho_0, us, xs, ts + dt_res, terminal_t, diffusion_eps, half_unroll_nums)
    dphidx = (phi_right - phi_left) / (2*dx_res)
    dphidt = (phi_up - phi_middle) / dt_res
    d2phidx2 = (phi_right - 2*phi_middle + phi_left) / (dx_res**2)
    HJ_res = dphidt - 0.5 * (dphidx**2) + 0.5 * diffusion_eps * d2phidx2

    vrho_left = -(phi_middle - phi_left) / dx_res * rho_left
    vrho_middle = -(phi_right - phi_middle) / dx_res * rho_middle
    dvrhodx = (vrho_middle - vrho_left) / dx_res
    drhodt = (rho_up - rho_middle) / dt_res
    d2rhodx2 = (rho_right - 2*rho_middle + rho_left) / (dx_res**2)
    cont_res = drhodt + dvrhodx - d2rhodx2 / 2 * diffusion_eps

    obj_val = jnp.sum(rho_middle * dphidx**2) /2
    return cont_res, HJ_res, obj_val


compute_mfc_residual_obj_periodic_batch = jax.vmap(compute_mfc_residual_obj_periodic, in_axes = (0, 0, None, None, None, None, None, None, None, None), out_axes = (0, 0, 0))
compute_mfc_residual_obj_periodic_batch = jax.jit(compute_mfc_residual_obj_periodic_batch, static_argnums=(9,))


def test_hj():
    diffusion_eps = 0.02
    seed = 1
    nx = 1600
    dx = 1.0 / nx
    t = 1.0
    dx_fd = 0.0003125
    # dt_fd = jnp.sqrt(dx_fd)
    dt_fd = dx_fd
    xs = jnp.linspace(0.0, 1.0, nx, endpoint=False) # (nx,)  domain of x [0,1] periodic

    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    gs = data_utils.generate_gaussian_process(next(rng), xs, 1, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
    gs = gs - (dx * jnp.sum(gs, axis = -1, keepdims = True)) # (1, nx)

    init_rho = data_utils.generate_gaussian_process(next(rng), xs, 1, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (1, nx)
    init_rho = jax.nn.softplus(init_rho) # (1, nx), positive
    init_rho = init_rho / (dx * jnp.sum(init_rho, axis = -1, keepdims = True)) # (1, nx), sum to 1

    # test viscous solver 
    half_repeat_num = 6
    g_u_repeat = extend_fn_periodic(gs[0,:], 2* half_repeat_num + 1)
    u_grids = jnp.linspace(0.0-half_repeat_num, 1.0 + half_repeat_num, nx * (2* half_repeat_num + 1), endpoint=False)
    phi = viscous_HJ_solver_1d_Riemann(g_u_repeat, u_grids, xs, 1.0 + 0.0* xs, diffusion_eps)
    phi_left = viscous_HJ_solver_1d_Riemann(g_u_repeat, u_grids, xs - dx_fd, 1.0 + 0.0* xs, diffusion_eps)
    phi_right = viscous_HJ_solver_1d_Riemann(g_u_repeat, u_grids, xs + dx_fd, 1.0 + 0.0* xs, diffusion_eps)
    phi_upper = viscous_HJ_solver_1d_Riemann(g_u_repeat, u_grids, xs, 1.0 + dt_fd + 0.0* xs, diffusion_eps)
    gradx_phi = viscous_HJ_gradx_1d_Riemann(g_u_repeat, u_grids, xs, 1.0 + 0.0* xs, diffusion_eps)
    dphi_dt = (phi_upper - phi) / dt_fd
    dphi_dx = (phi_right - phi_left) / (2*dx_fd)
    Lap_phi = (phi_right - 2*phi + phi_left) / (dx_fd**2)
    residual = dphi_dt + dphi_dx **2/2 - diffusion_eps * Lap_phi /2
    err_dphi_dx = jnp.abs(dphi_dx - gradx_phi)
    print('err dphi_dx max {}, mean {}'.format(jnp.max(err_dphi_dx), jnp.mean(err_dphi_dx)))
    print('residual max {}, mean {}'.format(jnp.max(jnp.abs(residual)), jnp.mean(jnp.abs(residual))))


def test_unbdd_zerog_Gaussianrho0():
    '''
    test case for mfc in unbdd domain R with g=0, rho0 = Gaussian(0.5, var = sigma_init = 0.001)
    true sol: HJ sol = 0, rho = Gaussian(0.5, var = sigma_t = sigma_init + t * diffusion_eps)
    Note: domain for x should be smaller than domain for u
    '''
    nu = 10000
    nx = 100
    diffusion_eps = 0.02
    t = 1.0
    terminal_t = 2.23
    u_left = -1.0
    u_right = 2.0
    x_left = 0.0
    x_right = 1.0
    us = jnp.linspace(u_left, u_right, nu) # (nu,)
    xs = jnp.linspace(x_left, x_right, nx) # (nx,)
    g = jnp.zeros_like(us)
    sigma_init = 0.001
    sigma_t = sigma_init + t * diffusion_eps
    rho_0 = jnp.exp(-0.5 * (us - 0.5)**2 / sigma_init) / (jnp.sqrt(2 * jnp.pi * sigma_init))
    # rho_0 = rho_0 / (sum(rho_0) * (us[1] - us[0]))
    rho_true = jnp.exp(-0.5 * (xs - 0.5)**2 / (sigma_t)) / (jnp.sqrt(2 * jnp.pi * (sigma_t)))
    phi_true = jnp.zeros_like(xs)
    # rho_true = rho_true / (sum(rho_true) * (xs[1] - xs[0]))
    rho_num, phi_num = solve_mfc_unbdd(g, us, rho_0, us, xs, t + 0*xs, terminal_t, diffusion_eps)
    # rho_num = rho_num / (sum(rho_num) * (xs[1] - xs[0]))
    max_rho_err = jnp.max(jnp.abs(rho_num - rho_true)) / jnp.max(rho_true)
    mean_rho_err = jnp.mean(jnp.abs(rho_num - rho_true)) / jnp.mean(rho_true)
    max_phi_err = jnp.max(jnp.abs(phi_num - phi_true))
    mean_phi_err = jnp.mean(jnp.abs(phi_num - phi_true))
    print('zero g, Gaussian rho0, rho err max {}, mean {}, phi err max {}, mean {}'.format(max_rho_err, mean_rho_err, max_phi_err, mean_phi_err))


def test_unbdd_zerog_unifrho0():
    '''
    test case for mfc in unbdd domain R with g=0, rho0 = uniform(y_left = -0.2, y_right = 0.5)
    true sol: HJ sol = 0, rho = (cdf_Gaussian(x-y_left) - cdf_Gaussian(x-y_right)) / (y_right - y_left)
    Note: domain for x should be smaller than domain for u
    '''
    nu = 10000
    nx = 100
    ny = 10000
    diffusion_eps = 0.02
    t = 1.0
    terminal_t = 2.23
    u_left = -5.0  # u is large enough, grid pts of g
    u_right = 3.0
    x_left = -4.0
    x_right = 2.0
    y_left = -0.2  # support of uniform rho0
    y_right = 0.5
    us = jnp.linspace(u_left, u_right, nu) # (nu,)
    xs = jnp.linspace(x_left, x_right, nx) # (nx,)
    ys = jnp.linspace(y_left, y_right, ny) # (ny,)
    g = jnp.zeros_like(us)
    rho_0 = jnp.ones_like(ys)
    rho_0 = rho_0 / (sum(rho_0) * (ys[1] - ys[0]))
    rho_true = jax.scipy.stats.norm.cdf(xs - y_left, loc=0.0, scale=jnp.sqrt(diffusion_eps * t)) -\
          jax.scipy.stats.norm.cdf(xs - y_right, loc=0.0, scale=jnp.sqrt(diffusion_eps * t))
    rho_true = rho_true / (y_right - y_left)
    phi_true = jnp.zeros_like(xs)
    # rho_true = rho_true / (sum(rho_true) * (xs[1] - xs[0]))
    rho_num, phi_num = solve_mfc_unbdd(g, us, rho_0, ys, xs, t + 0*xs, terminal_t, diffusion_eps, if_y_grids_full_domain = False)
    # rho_num = rho_num / (sum(rho_num) * (xs[1] - xs[0]))
    max_rho_err = jnp.max(jnp.abs(rho_num - rho_true)) / jnp.max(rho_true)
    mean_rho_err = jnp.mean(jnp.abs(rho_num - rho_true)) / jnp.mean(rho_true)
    max_phi_err = jnp.max(jnp.abs(phi_num - phi_true))
    mean_phi_err = jnp.mean(jnp.abs(phi_num - phi_true))
    print('zero g, unif rho0, rho err max {}, mean {}, phi err max {}, mean {}'.format(max_rho_err, mean_rho_err, max_phi_err, mean_phi_err))


def test_unbdd_quadg_Gaussianrho0():
    '''
    test case for mfc in unbdd domain R with g = (x-c_g)^2/2/sigma_g, rho0 = Gaussian(c_0, var = sigma_0)
    true sol:   HJ sol = (x-c_g)^2/2/(sigma_g + T-t) + eps/2*log(1+(T-t)/sigma_g)  [with terminal cond g]
                rho = Gaussian(c_t, var = sigma_t), where
                    sigma_t = (sigma_g + T-t)/(sigma_g + T)*(eps * t + sigma_0*(sigma_g + T-t)/(sigma_g + T))
                    c_t = c_g + (c_0-c_g)(sigma_g +T-t) / (sigma_g + T)
    '''
    nu = 10000
    ny = 10000
    nx = 100
    diffusion_eps = 0.02
    t = 0.5
    u_left = -6.0
    u_right = 6.0
    x_left = 0.0
    x_right = 0.6
    y_left = -2.0
    y_right = 3.0
    us = jnp.linspace(u_left, u_right, nu) # (nu,)
    xs = jnp.linspace(x_left, x_right, nx) # (nx,)
    ys = jnp.linspace(y_left, y_right, ny) # (ny,)
    c_g = (u_left + u_right) / 2
    sigma_g = 1.0
    terminal_t = sigma_g  # choose time carefully to avoid numerical blow-up
    g = (us - c_g)**2 / 2 / sigma_g
    sigma_0 = diffusion_eps * sigma_g  # choose sigma's carefully
    sigma_t = (sigma_g + terminal_t - t) / (sigma_g + terminal_t) * (diffusion_eps * t + sigma_0 * (sigma_g + terminal_t - t) / (sigma_g + terminal_t))
    c_0 = (y_left + y_right) / 2
    c_t = c_g + (c_0 - c_g) * (sigma_g + terminal_t - t) / (sigma_g + terminal_t)
    rho_0 = jnp.exp(-0.5 * (ys - c_0)**2 / sigma_0) / (jnp.sqrt(2 * jnp.pi * sigma_0))  # (ny,)
    # rho_0 = rho_0 / (sum(rho_0) * (ys[1] - ys[0]))
    rho_true = jnp.exp(-0.5 * (xs - c_t)**2 / (sigma_t)) / (jnp.sqrt(2 * jnp.pi * (sigma_t)))  # (nx,)
    # rho_true = rho_true / (sum(rho_true) * (xs[1] - xs[0]))
    phi_true = (xs - c_g)**2 / 2 / (sigma_g + terminal_t - t) + diffusion_eps / 2 * jnp.log(1 + (terminal_t - t) / sigma_g)
    sigma_fwd = 1/(diffusion_eps / sigma_0 - 1/ (sigma_g + terminal_t))
    c_fwd = (c_0*diffusion_eps / sigma_0 - c_g / (sigma_g + terminal_t)) * sigma_fwd
    print('sigma_fwd {}, c_fwd {}'.format(sigma_fwd, c_fwd))  # make sure domain of y covers important values of this quad fn
    rho_num, phi_num = solve_mfc_unbdd(g, us, rho_0, ys, xs, t + 0*xs, terminal_t, diffusion_eps)
    # rho_num = rho_num / (sum(rho_num) * (xs[1] - xs[0]))
    max_rho_err = jnp.max(jnp.abs(rho_num - rho_true)) / jnp.max(rho_true)
    mean_rho_err = jnp.mean(jnp.abs(rho_num - rho_true)) / jnp.mean(rho_true)
    max_phi_err = jnp.max(jnp.abs(phi_num - phi_true)) / jnp.max(jnp.abs(phi_true))
    mean_phi_err = jnp.mean(jnp.abs(phi_num - phi_true)) / jnp.mean(jnp.abs(phi_true))
    print('quad g, Gaussian rho0, rho err max {}, mean {}, phi err max {}, mean {}'.format(max_rho_err, mean_rho_err, max_phi_err, mean_phi_err))
    

def test_unbdd_quadg_truncatedGaussianrho0():
    '''
    test case for mfc in unbdd domain R with g = (x-c_g)^2/2/sigma_g, rho0 = pdf_Gaussian(c_g, var = eps(sigma_g + T)) * pdf_unif(y_left, y_right)/Z
    true sol:   HJ sol = (x-c_g)^2/2/(sigma_g + T-t) + eps/2*log(1+(T-t)/sigma_g)  [with terminal cond g]
                rho = pdf_Gaussian(c_g, var = eps(sigma_g + T-t)) * (cdf_Gaussian(x-y_left) - cdf_Gaussian(x-y_right))/ (y_right - y_left)/Z
    Note: domain for x should be bigger than domain for y but smaller than domain for u
    '''
    nu = 10000
    nx = 100
    ny = 10000
    diffusion_eps = 0.02
    t = 1.0
    terminal_t = 2.23
    u_left = -5.0  # u is large enough, grid pts of g
    u_right = 3.0
    x_left = -4.0
    x_right = 2.0
    y_left = -0.2  # support of uniform rho0
    y_right = 0.5
    us = jnp.linspace(u_left, u_right, nu) # (nu,)
    xs = jnp.linspace(x_left, x_right, nx) # (nx,)
    ys = jnp.linspace(y_left, y_right, ny) # (ny,)
    c_g = (u_left + u_right) / 2
    sigma_g = 1.0
    g = (us - c_g)**2 / 2 / sigma_g
    rho_0 = jnp.exp(-0.5 * (ys - c_g)**2 / (diffusion_eps * (sigma_g + terminal_t))) # (ny,)
    rho_0 = rho_0 / (sum(rho_0) * (ys[1] - ys[0]))
    rho_true = (jax.scipy.stats.norm.cdf(xs - y_left, loc=0.0, scale=jnp.sqrt(diffusion_eps * t)) -\
          jax.scipy.stats.norm.cdf(xs - y_right, loc=0.0, scale=jnp.sqrt(diffusion_eps * t))) * \
          jnp.exp(-0.5 * (xs - c_g)**2 / (sigma_g + terminal_t - t) / diffusion_eps)
    Z = sum(rho_true) * (xs[1] - xs[0])
    rho_true = rho_true / Z
    phi_true = (xs - c_g)**2 / 2 / (sigma_g + terminal_t - t) + diffusion_eps / 2 * jnp.log(1 + (terminal_t - t) / sigma_g)
    rho_num, phi_num = solve_mfc_unbdd(g, us, rho_0, ys, xs, t + 0*xs, terminal_t, diffusion_eps, if_y_grids_full_domain = False)
    rho_num = rho_num / (sum(rho_num) * (xs[1] - xs[0]))
    max_rho_err = jnp.max(jnp.abs(rho_num - rho_true)) / jnp.max(rho_true)
    mean_rho_err = jnp.mean(jnp.abs(rho_num - rho_true)) / jnp.mean(rho_true)
    max_phi_err = jnp.max(jnp.abs(phi_num - phi_true)) / jnp.max(jnp.abs(phi_true))
    mean_phi_err = jnp.mean(jnp.abs(phi_num - phi_true)) / jnp.mean(jnp.abs(phi_true))
    print('zero g, unif rho0, rho err max {}, mean {}, phi err max {}, mean {}'.format(max_rho_err, mean_rho_err, max_phi_err, mean_phi_err))
    

def test_bdd_randg_randrho0(nx, nt, nu_nx_ratio, if_plot = True):
    '''
    test case for periodic problem: compute residual errors and compare with optimization method
    @parameters:
        nx: int, number of grid points for x
        nt: int, number of grid points for t is nt+1 including t=0 and t=T
        nu_nx_ratio: int, nu/nx
        if_plot: bool, if plot the solution
    '''
    nu = nx * nu_nx_ratio
    diffusion_eps = 0.04
    seed = 1
    u_left = 0.0
    u_right = 1.0
    x_left = 0.0
    x_right = 1.0
    terminal_t = 1.0

    batch = 10

    dt = terminal_t / nt
    du = (u_right - u_left) / nu
    dx = (x_right - x_left) / nx
    print(f'dx = {dx}, du = {du}, dt = {dt}')
    timer.tic('datagen')
    us = jnp.linspace(u_left, u_right, nu, endpoint=False) # (nu,)
    xs = jnp.linspace(x_left, x_right, nx, endpoint=False) # (nx,)
    rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    # gs = data_utils.generate_gaussian_process(next(rng), us, batch, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (batch, nu)
    # gs = jax.nn.softplus(gs) # (batch, nu), positive
    # gs = -diffusion_eps * jnp.log(gs) # (batch, nu)
    # gs = gs - (du * jnp.sum(gs, axis = -1, keepdims = True)) # (batch, nu)
    gs = data_utils.generate_gaussian_process(next(rng), us, batch, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)
    gs = gs - jnp.mean(gs, axis = -1, keepdims = True) # (eqns, nu)
    gs = gs/20.0
    
    rho_0 = data_utils.generate_gaussian_process(next(rng), us, batch, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0) # (batch, nu)
    rho_0 = jax.nn.softplus(rho_0) # (batch, nu), positive
    rho_0 = rho_0 / (du * jnp.sum(rho_0, axis = -1, keepdims = True)) # (batch, nu), sum to 1
    
    xs_all = einshape('i->ij', xs, j = nt+1)  # (nx, nt+1)
    xs_all = einshape('ij->(ij)', xs_all)  # (n_pts), where n_pts = (nt+1) * nx
    ts_all = einshape('i->ji', jnp.linspace(0, terminal_t, nt+1), j = nx)  # (nx, nt+1)
    ts_all = einshape('ij->(ij)', ts_all)  # (n_pts), where n_pts = (nt+1) * nx
    gs_coarse = gs[:,0::nu_nx_ratio]
    rho0_coarse = rho_0[:,0::nu_nx_ratio]

    timer.toc('datagen')

    # solve it using optimization method for mfc
    mfc_config = mfc.get_config(nt, nx, diffusion_eps/2.0)  # NOTE: the parameter here is the coeff on Lap, hence eps/2
    run_cost_batch = jnp.ones((batch, nt*nx)) * 0.5 # fixed 
    mfc_iters = 1000
    mfc_tol = 1e-3
    mfc_verbose = False
    # warmup run
    x_batch, res_batch = mfc.forward_batch(nt, nx, mfc_config, rho0_coarse, gs_coarse, run_cost_batch, gamma = 0.2, 
                                              max_iters=mfc_iters, check_freq = 100, tol= mfc_tol, verbose=mfc_verbose)
    timer.tic('mfc_solver_1')
    x_batch, res_batch = mfc.forward_batch(nt, nx, mfc_config, rho0_coarse, gs_coarse, run_cost_batch, gamma = 0.2, 
                                              max_iters=mfc_iters, check_freq = 100, tol= mfc_tol, verbose=mfc_verbose)
    timer.toc('mfc_solver_1')
    print('mfc res batch {}'.format(jnp.max(res_batch)))
    assert jnp.all(res_batch < mfc_tol)
    rhos = einshape("i(mn)->imn", x_batch[:,:nt*nx], m = nt, n = nx) # (batch, nt, nx)
    rhos = einshape("imn->inm", rhos) # (batch, nx, nt)
    rhos = jnp.concatenate([rho0_coarse[:,:,None], rhos], axis = -1) # (batch, nx, nt+1)
    m1 = einshape("i(mn)->imn", x_batch[:,nt*nx:2*nt*nx], m = nt, n = nx) # (batch, nt, nx)
    m1 = einshape("imn->inm", m1) # (batch, nx, nt)
    m2 = einshape("i(mn)->imn", x_batch[:,2*nt*nx:], m = nt, n = nx) # (batch, nt, nx)
    m2 = einshape("imn->inm", m2) # (batch, nx, nt)
    obj_ver1 = jnp.sum(m1**2 / rhos[:,:,1:] + m2**2 * rhos[:,:,1:]) * dx * dt /2
    constraint = (rhos[:,:,1:] - rhos[:,:,:-1]) / dt + (m1 - jnp.roll(m1,1,1)) / dx + (jnp.roll(m2,-1,1) - m2) / dx
    constraint = constraint - diffusion_eps/2 * (jnp.roll(rhos[:,:,1:],-1,1) - 2*rhos[:,:,1:] + jnp.roll(rhos[:,:,1:],1,1)) / dx**2
    constraint_abs = jnp.abs(constraint)
    print('ver1 obj {}, constraint mean {}, constraint max {}'.format(obj_ver1, jnp.mean(constraint_abs), jnp.max(constraint_abs)))
    rho_ver1 = rhos

    # solve it using integration method
    half_unroll_nums = get_half_unroll_nums(us, us, ts_all, terminal_t, diffusion_eps)
    # warmup run
    rho_all, phi_all = solve_mfc_periodic_batch(gs, us, rho_0, us, xs_all, ts_all, terminal_t, diffusion_eps, half_unroll_nums)
    timer.tic('mfc_solver_2')
    rho_all, phi_all = solve_mfc_periodic_batch(gs, us, rho_0, us, xs_all, ts_all, terminal_t, diffusion_eps, half_unroll_nums)
    timer.toc('mfc_solver_2')
    rho_ver2 = einshape("i(mn)->imn", rho_all, m = nx, n = nt+1) # (batch, nx, nt+1)
    phi_ver2 = einshape("i(mn)->imn", phi_all, m = nx, n = nt+1) # (batch, nx, nt+1)
    # compute residual and obj val in mfc
    dx_res = 0.00001
    dt_res = dx_res
    x_all_respts = einshape('i->ij', xs, j = nt-1)  # (nx, nt-1)
    x_all_respts = einshape('ij->(ij)', x_all_respts)  # (n_pts), where n_pts = (nt-1) * nx
    t_all_respts = einshape('i->ji', jnp.linspace(terminal_t/nt, terminal_t, nt-1, endpoint=False), j = nx)  # (nx, nt-1)
    t_all_respts = einshape('ij->(ij)', t_all_respts)  # (n_pts), where n_pts = (nt-1) * nx
    res_cont, res_hj, obj_ver2 = compute_mfc_residual_obj_periodic_batch(gs, rho_0, us, x_all_respts, t_all_respts, terminal_t, 
                                                                         diffusion_eps, dx_res, dt_res, half_unroll_nums)
    res_cont_abs = jnp.abs(res_cont)
    res_hj_abs = jnp.abs(res_hj)
    obj_ver2 *= (dx*dt)
    print('ver2 obj {}, res cont mean {}, res cont max {}, res hj mean {}, res hj max {}'.format(obj_ver2, 
                    jnp.mean(res_cont_abs, axis = -1), jnp.max(res_cont_abs, axis = -1), jnp.mean(res_hj_abs, axis = -1), jnp.max(res_hj_abs, axis = -1)))

    # check errors between ver1 and ver2
    err = jnp.abs(rho_ver2 - rho_ver1)
    err_mean = jnp.mean(err) / jnp.mean(rho_ver1)
    err_max = jnp.max(err) / jnp.max(rho_ver1)
    print('rho err mean {}, max {}'.format(err_mean, err_max))

    # check velocity: compare the velocity of the following three functions: (they should be different but close)
    #   1. ver2 method at t = T - dt_small
    #   2. ver1 method at t = T - dt using (m1+m2)/rho
    #   3. the one computed using terminal g
    dx_small = 0.01
    dt_small = 0.01
    t = terminal_t - dt_small
    half_unroll_nums = get_half_unroll_nums(us, us, t + 0*xs, terminal_t, diffusion_eps)
    _, phi_left = solve_mfc_periodic_batch(gs, us, rho_0, us, xs - dx_small, t + 0*xs, terminal_t, diffusion_eps, half_unroll_nums)
    _, phi_right = solve_mfc_periodic_batch(gs, us, rho_0, us, xs + dx_small, t + 0*xs, terminal_t, diffusion_eps, half_unroll_nums)
    v_ver2 = -(phi_right - phi_left) / dx_small / 2
    gs_right = jnp.roll(gs, -1, axis = -1)
    v_last = -(gs_right[:, ::nu_nx_ratio] - gs[:, ::nu_nx_ratio]) / du
    v_ver1 = (m1+m2)[:, :,-1] / rhos[:, :,-1]
    print('velocity diff: v2 and v1 {}, v1 and vg {}, v2 and vg {}'.format(jnp.mean(jnp.abs(v_ver2 - v_ver1)),
                            jnp.mean(jnp.abs(v_ver1 - v_last)), jnp.mean(jnp.abs(v_last - v_ver2))), flush=True)
    
    if if_plot:
        for fi, (rho_0, gs, constraint, res_cont, res_hj, rho_ver1, rho_ver2, phi_ver2, v_ver1,  v_ver2, v_last) in enumerate(zip(rho_0, gs, constraint, res_cont, res_hj, rhos, rho_ver2, phi_ver2, v_ver1, v_ver2, v_last)):
            plt.close('all')
            fig = plt.figure()
            plt.plot(xs, constraint)
            fig.savefig(f'constraint_ver1_nx_{nx}_nt_{nt}_{fi}.png')

            fig = plt.figure()
            res_cont_reshape = einshape("(mn)->mn", res_cont, m = nx, n = nt-1)
            plt.plot(xs, res_cont_reshape)
            fig.savefig(f'res_cont_nx_{nx}_nt_{nt}_{fi}.png')

            fig = plt.figure()
            res_hj_reshape = einshape("(mn)->mn", res_hj, m = nx, n = nt-1)
            plt.plot(xs, res_hj_reshape)
            fig.savefig(f'res_hj_nx_{nx}_nt_{nt}_{fi}.png')

            fig = plt.figure()
            plt.plot(xs, v_ver2, label = 'v ver2')
            plt.plot(xs, v_ver1, label = 'v ver1')
            plt.plot(xs, v_last, label = 'v terminal cond')
            plt.legend()
            fig.savefig(f'test_velocity_nx_{nx}_nt_{nt}_{fi}.png')
            
            # residual_cont_2, residual_HJ_2 = compute_OT_residual(dx, dt, phi_num_all, rhos, diffusion_eps, periodic)
            # print('residual2 cont {}, HJ {}'.format(jnp.max(jnp.abs(residual_cont_2)), jnp.max(jnp.abs(residual_HJ_2))))

            fig = plt.figure()
            plot_last_index = nt
            plt.plot(xs, rho_ver1[:,:plot_last_index+1], ':', label='ver1')
            plt.plot(xs, rho_ver1[:,1], '*', label='ver1')
            plt.plot(xs, rho_ver2[:,:plot_last_index+1], label='ver2')
            plt.plot(xs, rho_ver2[:,1], '*', label='ver2')
            plt.plot(us, rho_0, 'k', label='init')
            fig.savefig("test_rho_nx_{}_nt_{}_{}.png".format(nx, nt, fi))

            fig = plt.figure()
            plot_last_index = nt
            err = (rho_ver2 - rho_ver1) / rho_ver1
            plt.plot(xs, err[:,1:plot_last_index+1])
            plt.plot(xs, err[:,1], '*')
            fig.savefig("test_rho_relerr_nx_{}_nt_{}_{}.png".format(nx, nt, fi))

            fig = plt.figure()
            plot_last_index = nt
            plt.plot(xs, phi_ver2[:,:plot_last_index+1], label='ver2')
            plt.plot(xs, phi_ver2[:,1], '*', label='ver2')
            plt.plot(us, gs, 'k', label='terminal g')
            fig.savefig("test_phi_nx_{}_nt_{}_{}.png".format(nx, nt, fi))


if __name__ == "__main__":
    import data_mfc_pdhg as mfc

    # test on hj utils
    print('===============Test on hj utils=================')
    test_hj()
    
    # test on R: all passed
    print('===============Test on R=================')
    print('----------------test 1-------------------')
    test_unbdd_zerog_Gaussianrho0()
    print('----------------test 2-------------------')
    test_unbdd_zerog_unifrho0()
    print('----------------test 3-------------------')
    test_unbdd_quadg_Gaussianrho0()
    print('----------------test 4-------------------')
    test_unbdd_quadg_truncatedGaussianrho0()
    
    # test in periodic domain [0,1]
    print('===============Test on periodic domain [0,1] =================')
    # print('---------------nx=100, nt=20, nu=1000------------------')
    # test_bdd_randg_randrho0(nx = 100, nt = 20, nu_nx_ratio=10, if_plot = True)
    print('---------------nx=100, nt=20, nu=100------------------')
    test_bdd_randg_randrho0(nx = 100, nt = 20, nu_nx_ratio=1, if_plot = True)