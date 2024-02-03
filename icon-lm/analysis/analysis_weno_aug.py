
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax
import jax.numpy as jnp
from jax.config import config
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from collections import OrderedDict

import sys
sys.path.append('../data_preparation/')
sys.path.append('../')
sys.path.append('../data_preparation/weno/')
import utils
from datagen_weno import generate_weno_scalar_sol


def write_quadratic_consistency_error(folder, eqn_name):
    with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
    consistency_dict = {}
    for key in result_dict.keys():
        if key[0] == eqn_name and key[3] == 'pred':
            print(key, flush = True)
            _, coeff_a, coeff_b, _, demo_num, caption_id = key
            fn = jax.jit(lambda u: coeff_a * u * u + coeff_b * u)
            grad_fn = jax.jit(lambda u: 2 * coeff_a * u + coeff_b)
            forward = generate_weno_scalar_sol(dx = 0.01, dt = 0.001, init = result_dict[key], fn = fn, steps = 100, grad_fn = grad_fn)[:,-1,...]
            consistency_dict[(eqn_name, coeff_a, coeff_b, 'forward', demo_num, caption_id)] = forward                
    # save consistency_dict to the same folder
    with open("{}/consistency_dict.pkl".format(folder), "wb") as file:
        pickle.dump(consistency_dict, file)


def write_cubic_consistency_error(folder, eqn_name):
    with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
    consistency_dict = {}
    for key in result_dict.keys():
        if key[0] == eqn_name and key[4] == 'pred':
            print(key, flush = True)
            _, coeff_a, coeff_b, coeff_c, _, demo_num, caption_id = key
            fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
            grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
            forward = generate_weno_scalar_sol(dx = 0.01, dt = 0.0005, init = result_dict[key], fn = fn, steps = 200, grad_fn = grad_fn)[:,-1,...]
            consistency_dict[(eqn_name, coeff_a, coeff_b, coeff_c, 'forward', demo_num, caption_id)] = forward                
    # save consistency_dict to the same folder
    with open("{}/consistency_dict.pkl".format(folder), "wb") as file:
        pickle.dump(consistency_dict, file)

def write_sin_consistency_error(folder, eqn_name):
    with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
    consistency_dict = {}
    for key in result_dict.keys():
        if key[0] == eqn_name and key[5] == 'pred':
            print(key, flush = True)
            _, coeff_a, coeff_b, coeff_c, stride, _, demo_num, caption_id = key
            fn = jax.jit(lambda u: coeff_a * jnp.sin(coeff_c * u) + coeff_b * jnp.cos(coeff_c * u))
            grad_fn = jax.jit(lambda u: coeff_a * coeff_c + jnp.cos(coeff_c * u) - coeff_b * coeff_c + jnp.sin(coeff_c * u))
            forward = generate_weno_scalar_sol(dx = 0.01, dt = 0.0005, init = result_dict[key], fn = fn, steps = int(stride), grad_fn = grad_fn)[:,-1,...]
            consistency_dict[(eqn_name, coeff_a, coeff_b, coeff_c, stride, 'forward', demo_num, caption_id)] = forward                
    # save consistency_dict to the same folder
    with open("{}/consistency_dict.pkl".format(folder), "wb") as file:
        pickle.dump(consistency_dict, file)

if __name__ == "__main__":

    # folder = "/home/shared/icon/analysis/icon_weno_20230829-170831_light"
    # eqn_name = 'conservation_weno_quadratic_backward'
    # write_consistency_error(folder, eqn_name)

    folder = "/home/shared/icon/analysis/icon_weno_20230904-184910_sin"
    eqn_name = 'conservation_weno_sin_backward'
    write_sin_consistency_error(folder, eqn_name)
