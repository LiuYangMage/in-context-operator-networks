
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import matplotlib.patches as patches

cmap_1 = 'Blues'
cmap_2 = 'Reds'
# cmap_1 = 'bwr'


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


def get_error_from_dict(result_dict, key, demo_num):
    error, relative_error = calculate_error(result_dict[(*key, 'pred', demo_num, -1)], 
                                            result_dict[(*key, 'ground_truth')], result_dict[(*key, 'mask')])
    return error, relative_error

def get_consistency_error_from_dict(result_dict, consistency_dict, key, demo_num):
   forward = consistency_dict[(*key, 'forward', demo_num, -1)]
   error, relative_error = calculate_error(forward, result_dict[(*key, 'cond_v')], result_dict[(*key, 'cond_mask')])
   return error, relative_error

def get_violation_from_dict(result_dict, key, demo_num):
    violation = calculate_conservation_violation(result_dict[(*key, 'pred', demo_num, -1)], 
                                                 result_dict[(*key, 'ground_truth')], result_dict[(*key, 'mask')])
    return violation


def get_matrix(result_dict, consistency_dict, a_range, b_range, c_range, eqn_name, demo_num, mode):
  error_matrix = np.zeros((len(a_range), len(b_range),len(c_range)))
  for i, coeff_a in enumerate(a_range):
      for j, coeff_b in enumerate(b_range):
        for k, coeff_c in enumerate(c_range):
          # round to 0.
          coeff_a = round(coeff_a, 1)
          coeff_b = round(coeff_b, 1)
          coeff_c = round(coeff_c, 1)
          key = (eqn_name, coeff_a, coeff_b, coeff_c)
          if mode == 'error':
            error_matrix[i, j] = get_error_from_dict(result_dict, key, demo_num)[0]
          elif mode == 'rerror':
            error_matrix[i, j] = get_error_from_dict(result_dict, key, demo_num)[1]
          elif mode == 'violation':
            error_matrix[i, j] = get_violation_from_dict(result_dict, key, demo_num)
          elif mode == 'consist':
            error_matrix[i, j] = get_consistency_error_from_dict(result_dict, consistency_dict, key, demo_num)[0]
          elif mode == 'rconsist':
            error_matrix[i, j] = get_consistency_error_from_dict(result_dict, consistency_dict, key, demo_num)[1]
          else:
            raise NotImplementedError
  return error_matrix

def get_average_error_list(result_dict, consistency_dict, a_range, b_range, c_range, eqn_name, mode, demo_num_list):
  errors = []
  for demo_num in demo_num_list:
    this_error = get_matrix(result_dict, consistency_dict, a_range, b_range, c_range, eqn_name, demo_num, mode)
    errors.append(np.mean(this_error))
  return errors
  
def draw_decay(a_range, b_range, c_range, demo_num_list):

    plt.figure(figsize=(5, 4))

    folders = ["/home/shared/icon/analysis/icon_weno_20231213-111526",
               "/home/shared/icon/analysis/icon_weno_20231209-222440",
               "/home/shared/icon/analysis/icon_weno_20231217-122838"]
    labels = ["consistency loss, bs = 4 + 4", "$L_2$ loss, bs = 8 + 8", "$L_2$ loss, bs = 4 + 4"]
    forward_markers = ['bo--','ks--','r^--']
    backward_markers = forward_markers

    plt.figure(figsize=(4, 3))
    for folder, label, forward_marker, backward_marker in zip(folders, labels, forward_markers, backward_markers):
      with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
      with open("{}/consistency_dict.pkl".format(folder), "rb") as file:
        consistency_dict = pickle.load(file)
      errors = get_average_error_list(result_dict, consistency_dict, a_range, b_range, c_range, 
                                      'conservation_weno_cubic_forward', 'error', demo_num_list)

      plt.plot(demo_num_list, errors, forward_marker, label = f'{label}')

    plt.xlabel('# examples')
    plt.xticks(demo_num_list)
    plt.ylabel('average error')
    plt.legend()
    plt.title('Forward Error')
    plt.tight_layout()
    plt.savefig(f'{folder}/decay_forward.pdf')
    plt.close('all')


    plt.figure(figsize=(4, 3))
    for folder, label, forward_marker, backward_marker in zip(folders, labels, forward_markers, backward_markers):
      with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
      with open("{}/consistency_dict.pkl".format(folder), "rb") as file:
        consistency_dict = pickle.load(file)
      consist = get_average_error_list(result_dict, consistency_dict, a_range, b_range, c_range,
                                      'conservation_weno_cubic_backward', 'consist', demo_num_list)
      plt.plot(demo_num_list, consist, backward_marker, label = f'{label}')
    plt.xlabel('# examples')
    plt.xticks(demo_num_list)
    plt.ylabel('average error')
    plt.legend()
    plt.title('Reverse Error')
    plt.tight_layout()
    plt.savefig(f'{folder}/decay_reverse.pdf')
    plt.close('all')

if __name__ == "__main__":
  demo_num_list = [1,2,3,4,5]

  a_range = np.linspace(-1, 1, 11)
  b_range = np.linspace(-1, 1, 11)
  c_range = np.linspace(-1, 1, 11)

  draw_decay(a_range = a_range, b_range=b_range, c_range=c_range, demo_num_list=demo_num_list)

