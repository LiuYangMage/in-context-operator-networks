
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import jax
from jax.config import config
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from collections import OrderedDict

def calculate_error(pred, label, mask, order = 1):
    '''
    pred: [batch, len, 1]
    label: [batch, len, 1]
    mask: [batch, len]
    '''
    error = np.mean(np.abs(pred - label))
    gt_norm_mean = np.mean(np.abs(label))
    relative_error = error/gt_norm_mean
    return error, relative_error

def calculate_conservation_violation(pred, label, mask):
    '''
    pred: [batch, len, 1]
    label: [batch, len, 1]
    mask: [batch, len]
    '''
    mask = mask.astype(bool)
    pred_mean = np.mean(pred, axis = -2) # [batch, 1]
    label_mean = np.mean(label, axis = -2)  # [batch, 1]
    violation = np.abs(pred_mean - label_mean)  # [batch, 1]
    violation = np.mean(violation)
    return violation


def get_error_from_dict(result_dict, key, demo_num, caption_id):
    error, relative_error = calculate_error(result_dict[(*key, 'pred', demo_num, caption_id)], result_dict[(*key, 'ground_truth')], result_dict[(*key, 'mask')])
    return error, relative_error

def get_consistency_error_from_dict(result_dict, consistency_dict, key, demo_num, caption_id):
   forward = consistency_dict[(*key, 'forward', demo_num, caption_id)]
   error, relative_error = calculate_error(forward, result_dict[(*key, 'cond_v')], result_dict[(*key, 'cond_mask')])
   return error, relative_error

def get_violation_from_dict(result_dict, key, demo_num, caption_id):
    violation = calculate_conservation_violation(result_dict[(*key, 'pred', demo_num, caption_id)], result_dict[(*key, 'ground_truth')], result_dict[(*key, 'mask')])
    return violation


cmap_1 = 'Blues'
cmap_2 = 'Reds'
# cmap_1 = 'bwr'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import matplotlib.patches as patches

def draw_error_in_one(folder, a_range, b_range, coeff_c, caption_id, demo_num, vmax_1, vmax_2):
    with open(f"{folder}/result_dict.pkl", "rb") as file:
        result_dict = pickle.load(file)

    with open(f"{folder}/consistency_dict.pkl", "rb") as file:
        consistency_dict = pickle.load(file)

    def get_matrix(eqn_name, mode):
      error_matrix = np.zeros((len(a_range), len(b_range)))
      for i, coeff_a in enumerate(a_range):
          for j, coeff_b in enumerate(b_range):
              coeff_a = round(coeff_a, 1)
              coeff_b = round(coeff_b, 1)
              key = (eqn_name, coeff_a, coeff_b, coeff_c)
              if mode == 'rerror':
                error_matrix[i, j] = get_error_from_dict(result_dict, key, demo_num, caption_id)[1]
              elif mode == 'consistency':
                error_matrix[i, j] = get_consistency_error_from_dict(result_dict, consistency_dict, key, demo_num, caption_id)[1]
              else:
                raise NotImplementedError
      return error_matrix
    
    a, b = np.meshgrid(a_range, b_range)

    # Create a grid layout
    fig = plt.figure(figsize=(10, 3))
    
    gs = gridspec.GridSpec(1, 5, width_ratios=[0.1, 1, 1, 1, 0.1])
    # gs.update(wspace=0.3)  # Reduce this value to bring subplots closer

    ax_cb1 = plt.subplot(gs[0])  # for the first colorbar
    ax0 = plt.subplot(gs[1])
    ax1 = plt.subplot(gs[2])
    ax2 = plt.subplot(gs[3])
    ax_cb2 = plt.subplot(gs[4])  # for the second colorbar

    # First subplot
    error_matrix = get_matrix("conservation_weno_cubic_backward", 'rerror')
    cax0 = ax0.pcolormesh(a, b, error_matrix.T, cmap=cmap_1, vmin=0, vmax=vmax_1)
    ax0.set_title('backward')
    ax0.set_xlabel(r'$a$')
    ax0.set_ylabel(r'$b$')
    ax0.set_aspect('equal', 'box')
    
    # First colorbar on the left
    cb1 = plt.colorbar(cax0, cax=ax_cb1)
    ax_cb1.yaxis.set_ticks_position('left')
    
    # Second subplot
    error_matrix = get_matrix("conservation_weno_cubic_backward", 'consistency')
    cax1 = ax1.pcolormesh(a, b, error_matrix.T, cmap=cmap_2, vmin=0, vmax=vmax_2)
    ax1.set_title('consistency')
    ax1.set_aspect('equal', 'box')
    # ax1.set_xlabel(r'$a$')
    # ax1.set_ylabel(r'$b$')
    # ax1.set_xticks([])
    # ax1.set_yticks([])
       
    # Third subplot
    error_matrix = get_matrix("conservation_weno_cubic_forward", 'rerror')
    cax2 = ax2.pcolormesh(a, b, error_matrix.T, cmap=cmap_2, vmin=0, vmax=vmax_2)
    ax2.set_title('forward')
    ax2.set_aspect('equal', 'box')
    # ax2.set_xlabel(r'$a$')
    # ax2.set_ylabel(r'$b$')
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # Second colorbar on the right
    cb2 = plt.colorbar(cax1, cax=ax_cb2)

    if np.min(a_range)!= -1 or np.max(a_range)!= 1 or np.min(b_range)!= -1 or np.max(b_range)!= 1:
      for ax in [ax0, ax1, ax2]:
        ax.add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='red', facecolor='none'))

    plt.suptitle(f'relative error, # examples = {demo_num}, c = {coeff_c}')
    plt.tight_layout()
    plt.savefig(f'{folder}/error_in_one_c_{coeff_c}_capid_{caption_id}_demonum_{demo_num}_err.pdf')


def draw_violation_in_one(folder, a_range, b_range, coeff_c, caption_id, demo_num, vmax_1, vmax_2):
    with open(f"{folder}/result_dict.pkl", "rb") as file:
        result_dict = pickle.load(file)

    def get_matrix(eqn_name):
        error_matrix = np.zeros((len(a_range), len(b_range)))
        for i, coeff_a in enumerate(a_range):
            for j, coeff_b in enumerate(b_range):
                # round to 0.1
                coeff_a = round(coeff_a, 1)
                coeff_b = round(coeff_b, 1)
                key = (eqn_name, coeff_a, coeff_b, coeff_c)
                error_matrix[i, j] = get_violation_from_dict(result_dict, key, demo_num, caption_id)
        return error_matrix

    a, b = np.meshgrid(a_range, b_range)

    # Create a grid layout
    fig = plt.figure(figsize=(8, 3))
    
    gs = gridspec.GridSpec(1, 4, width_ratios=[0.1, 1, 1, 0.1])
    # gs.update(wspace=0.3)  # Reduce this value to bring subplots closer

    ax_cb1 = plt.subplot(gs[0])  # for the first colorbar
    ax0 = plt.subplot(gs[1])
    ax1 = plt.subplot(gs[2])
    ax_cb2 = plt.subplot(gs[3])  # for the second colorbar

    # First subplot
    error_matrix = get_matrix("conservation_weno_cubic_backward")
    print(np.max(error_matrix), np.mean(error_matrix), np.min(error_matrix))
    cax0 = ax0.pcolormesh(a, b, error_matrix.T, cmap=cmap_1, vmin=0, vmax=vmax_1)
    ax0.set_title('backward')
    ax0.set_xlabel(r'$a$')
    ax0.set_ylabel(r'$b$')
    ax0.set_aspect('equal', 'box')

    # Second subplot
    error_matrix = get_matrix("conservation_weno_cubic_forward")
    print(np.max(error_matrix), np.mean(error_matrix), np.min(error_matrix))
    cax1 = ax1.pcolormesh(a, b, error_matrix.T, cmap=cmap_2, vmin=0, vmax=vmax_2)
    ax1.set_title('forward')
    # ax1.set_xlabel(r'$a$')
    # ax1.set_ylabel(r'$b$')
    ax1.set_aspect('equal', 'box')

    cb1 = plt.colorbar(cax0, cax=ax_cb1)
    ax_cb1.yaxis.set_ticks_position('left')
    cb2 = plt.colorbar(cax1, cax=ax_cb2)

    if np.min(a_range)!= -1 or np.max(a_range)!= 1 or np.min(b_range)!= -1 or np.max(b_range)!= 1:
      for ax in [ax0, ax1]:
        ax.add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='red', facecolor='none'))

    plt.suptitle(f'conservation violation, # examples = {demo_num}, c = {coeff_c}')
    plt.tight_layout()
    plt.savefig(f'{folder}/violation_in_one_c_{coeff_c}_capid_{caption_id}_demonum_{demo_num}_err.pdf')
    plt.close('all')


def draw_decay(folder, a_range, b_range, c_range, caption_id, demo_num_list):
    
    with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
    with open("{}/consistency_dict.pkl".format(folder), "rb") as file:
        consistency_dict = pickle.load(file)

    def get_average_error(eqn_name, demo_num, mode):
      error_matrix = np.zeros((len(a_range), len(b_range),len(c_range)))
      for i, coeff_a in enumerate(a_range):
          for j, coeff_b in enumerate(b_range):
            for k, coeff_c in enumerate(c_range):
              # round to 0.
              coeff_a = round(coeff_a, 1)
              coeff_b = round(coeff_b, 1)
              coeff_c = round(coeff_c, 1)
              key = (eqn_name, coeff_a, coeff_b, coeff_c)
              if mode == 'rerror':
                error_matrix[i, j] = get_error_from_dict(result_dict, key, demo_num, caption_id)[1]
              elif mode == 'violation':
                error_matrix[i, j] = get_violation_from_dict(result_dict, key, demo_num, caption_id)
              elif mode == 'consistency':
                error_matrix[i, j] = get_consistency_error_from_dict(result_dict, consistency_dict, key, demo_num, caption_id)[1]
              else:
                raise NotImplementedError
      return np.mean(error_matrix)

    def get_average_error_list(eqn_name, demo_num_list, mode):
      errors = []
      for demo_num in demo_num_list:
        errors.append(get_average_error(eqn_name, demo_num, mode))
      return errors
    
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.plot(demo_num_list, get_average_error_list('conservation_weno_cubic_backward', demo_num_list, 'rerror'), 'bo-', label = 'backward')
    plt.plot(demo_num_list, get_average_error_list('conservation_weno_cubic_backward', demo_num_list, 'consistency'), 'b^--', label = 'consistency')
    plt.plot(demo_num_list, get_average_error_list('conservation_weno_cubic_forward', demo_num_list, 'rerror'), 'ro-', label = 'forward')
    
    plt.xlabel('# examples')
    plt.ylabel('average relative error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(demo_num_list, get_average_error_list('conservation_weno_cubic_backward', demo_num_list, 'violation'), 'bo-', label = 'backward')
    plt.plot(demo_num_list, get_average_error_list('conservation_weno_cubic_forward', demo_num_list, 'violation'), 'ro-', label = 'forward')
    plt.xlabel('# examples')
    plt.ylabel('average conservation violation')
    plt.legend()

    plt.tight_layout()
    plt.savefig('{}/decay_capid_{}_err.pdf'.format(folder, caption_id))
    plt.close('all')

def draw_profile(eqn_name, a, b, c, example_num, caption_id, num_1, num_2):
    with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
    with open("{}/consistency_dict.pkl".format(folder), "rb") as file:
        consistency_dict = pickle.load(file)

    x = result_dict[(eqn_name, a, b, c, 'cond_k')]
    cond = result_dict[(eqn_name, a, b, c, 'cond_v')]
    qoi = result_dict[(eqn_name, a, b, c, 'ground_truth')]
    pred = result_dict[(eqn_name, a, b, c, 'pred', example_num, caption_id)]
    if 'backward' in eqn_name:
      forward = consistency_dict[(eqn_name, a, b, c, 'forward', example_num, caption_id)]
    print("x.shape", x.shape, "cond.shape", cond.shape, "qoi.shape", qoi.shape, "pred.shape", pred.shape)
    plt.figure(figsize=(num_2 * 4, num_1 * 3))
    for i in range(num_1 * num_2):
      plt.subplot(num_1, num_2, i+1)
      plt.plot(x[i,:,-1], cond[i,:,0], 'b-', label = 'condition')
      plt.plot(x[i,:,-1], qoi[i,:,0], 'k-', label = 'ground truth')
      plt.plot(x[i,:,-1], pred[i,:,0], 'r--', label = 'prediction')
      if 'backward' in eqn_name:
        plt.plot(x[i,:,-1], forward[i,:,0], 'g--', label = 'forward simulation')
      if i == 0:
        plt.legend()
      plt.xlabel('x')
      plt.ylabel('u')
    plt.tight_layout()
    plt.savefig('{}/profile_{}_{}_{}_{}_demo_{}_capid_{}_err.pdf'.format(folder, eqn_name, a, b, c, example_num, caption_id))
    plt.close('all')

if __name__ == "__main__":
  demo_num_list = [1,2,3,4,5]
  caption_id = -1

  folder = "/home/shared/icon/analysis/icon_weno_20231213-111526"
  a_range = np.linspace(-1, 1, 11)
  b_range = np.linspace(-1, 1, 11)
  c_range = np.linspace(-1, 1, 11)

  draw_decay(folder = folder, a_range = a_range, b_range=b_range, c_range=c_range, caption_id=caption_id, demo_num_list=demo_num_list)
  for demo_num in [demo_num_list[-1]]:
    for coeff_c in [-1, 0, 1]:
      draw_error_in_one(folder = folder, a_range=a_range, b_range=b_range, coeff_c = coeff_c, caption_id=caption_id, demo_num=demo_num, vmax_1=0.05, vmax_2=0.03)
      draw_violation_in_one(folder = folder, a_range=a_range, b_range=b_range, coeff_c = coeff_c, caption_id=caption_id, demo_num=demo_num, vmax_1 = 0.005, vmax_2=0.003)

  for coeff_a in [-0.6,0.6]:
    for coeff_b in [-0.6,0.6]:
      for coeff_c in [-0.6,0.6]:
        draw_profile(eqn_name = 'conservation_weno_cubic_backward', a = coeff_a, b = coeff_b, c = coeff_c, example_num = 5, caption_id = -1, num_1 = 2, num_2 = 5)
        draw_profile(eqn_name = 'conservation_weno_cubic_forward', a = coeff_a, b = coeff_b, c = coeff_c, example_num = 5, caption_id = -1, num_1 = 2, num_2 = 5)

