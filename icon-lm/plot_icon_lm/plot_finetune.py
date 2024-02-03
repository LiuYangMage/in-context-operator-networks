import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from absl import app, flags, logging
import sys
sys.path.append('../')

from jax.config import config
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
import utils
from plot import get_plot_k_index
import jax.numpy as jnp
import jax
from plot_utils import laplace_u_batch, calculate_error

shading = 'gouraud'

def pattern_match(patterns, name):
  for pattern in patterns:
    if pattern in name:
      return True
  return False

def sort_k_v(key, value, k_dim, k_mode, x_size):
  '''
  key: [size, k_dim]
  value: [size, x_size]
  return time, space, value
  '''
  k_v = np.concatenate([key, value], axis=-1) #[tx_size, kdim + 1]
  if k_mode == 'naive':
    sorted_indices = np.lexsort((k_v[:,1],k_v[:,0])) # time (0) first, then space (1)
    k_v = k_v[sorted_indices].reshape((-1, x_size, k_dim + 1))
    return k_v[...,0], k_v[...,1], k_v[...,-1]
  elif k_mode == 'itx':
    sorted_indices = np.lexsort((k_v[:,2],k_v[:,1])) # time (1) first, then space (2)
    k_v = k_v[sorted_indices].reshape((-1, x_size, k_dim + 1))
    return k_v[...,1], k_v[...,2], k_v[...,-1]

def get_ylim(data_list):
  min_max = np.array([[np.min(demo), np.max(demo)] for demo in data_list])
  ymin = np.min(min_max[:,0])
  ymax = np.max(min_max[:,1])
  gap = (ymax - ymin)
  return ymin - gap * 0.1, ymax + gap * 0.45    

def get_pretrain_result_dict(stamp, eqn_name):
  with open("{}/{}/deepo_pretrain_result_dict.pkl".format(FLAGS.analysis_folder, stamp), 'rb') as file:
    pretrain_result_dict = pickle.load(file)
    pretrain_pred = pretrain_result_dict[(eqn_name, 'pred', 5, -1)] # (bs, 101, 1)
  return pretrain_pred

def get_tuned_result_dict(stamp, eqn_name):

  deepo_tune_loss = []
  deepo_tune_pred = []
  deepo_tune_error = []
  for start, end in zip([0,100,200,300,400], [100,200,300,400,500]):
    with open("{}/{}/deepo_tune_result_dict_{}_{}.pkl".format(FLAGS.analysis_folder, stamp, start, end), 'rb') as file:
      this_result_dict = pickle.load(file)
      deepo_tune_loss.append(this_result_dict[(eqn_name, 'loss', 5, -1)]) # (bs, N)
      deepo_tune_pred.append(this_result_dict[(eqn_name, 'pred', 5, -1)]) # (bs, N, 101, 1)
      deepo_tune_error.append(this_result_dict[(eqn_name, 'error', 5, -1)]) # (bs, N,)
  deepo_tune_loss = np.concatenate(deepo_tune_loss, axis = 0) # (bs, N)
  deepo_tune_pred = np.concatenate(deepo_tune_pred, axis = 0) # (bs, N, 101, 1)
  deepo_tune_error = np.concatenate(deepo_tune_error, axis = 0) # (bs, N)
  
  # print("deepo loss:")
  # print(np.max(deepo_tune_loss, axis = 0), "\n", np.mean(deepo_tune_loss, axis = 0))

  # print("deepo tune pred error")
  # print(np.max(deepo_tune_error, axis = 0), "\n", np.mean(deepo_tune_error, axis = 0), "\n", np.std(deepo_tune_error, axis = 0))

  return deepo_tune_loss, deepo_tune_pred, deepo_tune_error

def get_data(eqn_name):
  '''
  get basic data for plotting
  '''

  with open("{}/{}/result_dict.pkl".format(FLAGS.analysis_folder, FLAGS.icon_stamp), 'rb') as file:
    result_dict = pickle.load(file)

  deepo_pretrain_pred = get_pretrain_result_dict(FLAGS.tune_deepo_stamp, eqn_name)
  fno_pretrain_pred = get_pretrain_result_dict(FLAGS.tune_fno_stamp, eqn_name)

  _, deepo_tune_pred, _ = get_tuned_result_dict(FLAGS.tune_deepo_stamp, eqn_name)
  _, fno_tune_pred, _ = get_tuned_result_dict(FLAGS.tune_fno_stamp, eqn_name)


  return (result_dict, 
          deepo_pretrain_pred, deepo_tune_pred,
          fno_pretrain_pred, fno_tune_pred)

def plot_error(eqn_name, result_dict, deepo_tune_pred, fno_tune_pred):

    ground_truth = result_dict[(eqn_name, 'ground_truth')] # (bs, 101, 1)
    icon_pred = result_dict[(eqn_name, 'pred', 5, -1)] # (bs, 101, 1)
    mask = np.ones_like(ground_truth)[...,0] # (bs, 101)
    
    icon_error, icon_relative_error = calculate_error(icon_pred, ground_truth, mask)

  
    last_idx = FLAGS.tune_steps // FLAGS.tune_record_freq

    deepo_errors = [calculate_error(deepo_tune_pred[:,t,...], ground_truth, mask) for t in range(last_idx+1)]
    deepo_errors = np.array(deepo_errors) # (N, 2), error, relative error
    
    fno_errors = [calculate_error(fno_tune_pred[:,t,...], ground_truth, mask) for t in range(last_idx+1)]
    fno_errors = np.array(fno_errors) # (N, 2), error, relative error

    plt.figure(figsize=(3.5, 2.5))
    ts = np.arange(0, FLAGS.tune_steps+1, FLAGS.tune_record_freq)
    
    # plt.plot(ts[0], deepo_errors[0,1], 'g*', label = "pretrained DeepONet")
    # plt.plot(ts[0], fno_errors[0,1], 'g*', label = "pretrained FNO")
    plt.plot(ts, deepo_errors[:,1], 'b-', label = "fine-tuning DeepONet")
    plt.plot(ts, fno_errors[:,1], 'g-', label = "fine-tuning FNO")

    plt.hlines(icon_relative_error, 0, FLAGS.tune_steps, 
               colors = 'r', linestyles = '--', label = "ICON-LM\nwithout fine-tuning")
    plt.xlabel('fine-tuning steps')
    plt.ylabel('relative error')
    plt.ylim(0, 0.3)
    plt.grid(True, which='both', axis='both', linestyle=':')
    plt.legend(loc = 'upper right', fontsize = 10) # 10 is the default
    plt.tight_layout()
    plt.savefig('{}/{}/{}_finetune_error.pdf'.format(FLAGS.analysis_folder, FLAGS.icon_stamp, eqn_name))
    print('save to {}/{}/{}_finetune_error.pdf'.format(FLAGS.analysis_folder, FLAGS.icon_stamp, eqn_name))
    
def plot_profile_full(eqn_name, result_dict, pretrain_pred, tune_pred, case_idx_list):
    
    x = result_dict[(eqn_name, 'cond_k')] # (bs, 101, 3), bs = 500
    cond = result_dict[(eqn_name, 'cond_v')] # (bs, 101, 1)
    qoi = result_dict[(eqn_name, 'ground_truth')] # (bs, 101, 1)
    pred = result_dict[(eqn_name, 'pred', 5, -1)] # (bs, 101, 1)
    demo_cond_v = result_dict[(eqn_name, 'demo_cond_v')] # (bs, 5, 101, 1)
    demo_qoi_v = result_dict[(eqn_name, 'demo_qoi_v')] # (bs, 5, 101, 1)
    
    uxx = laplace_u_batch(cond[...,0], 0.01) # (bs, 101)
    demo_num = demo_cond_v.shape[1]
    for case_idx in case_idx_list:
      plt.figure(figsize=(2 * 3.5, 2.5)) # rows, 2 columns
      plt.subplot(1, 2, 1) # condition
      for i in range(demo_num):
        plt.plot(x[case_idx,:,-1], demo_cond_v[case_idx, i,:,0], linestyle = ':', alpha = 0.8)
      plt.plot(x[case_idx,:,-1], cond[case_idx,:,0], 'k-', label = "question condition")
      plt.xlabel('$x$')
      plt.ylabel('$u$')
      ymin, ymax = get_ylim([demo_cond_v[case_idx], cond[case_idx]])
      plt.ylim(ymin, ymax)
      plt.legend(loc = 'upper right', fontsize = 8)
      plt.title('Condition')

      plt.subplot(1, 2, 2) # QoI
      for i in range(demo_num):
        plt.plot(x[case_idx,:,-1], demo_qoi_v[case_idx, i,:,0], linestyle = ':', alpha = 0.8)
      
      # random opearators
      for t in range(30):
        a = np.random.uniform(0.5, 1.5)
        k = np.random.uniform(0.5, 1.5)
        lamda = 0.1
        c = - lamda * a * uxx[case_idx] + k * cond[case_idx,:,0] ** 3
        plt.plot(x[case_idx,:,-1], c, 'k-', alpha = 0.1)
        
      plt.plot(x[case_idx,:,-1], qoi[case_idx,:,0], 'k-', label = "ground truth")
      plt.plot(x[case_idx,:,-1], pred[case_idx,:,0], 'r--', label = "ICON-LM")

      plt.plot(x[case_idx,:,-1], pretrain_pred[case_idx,:,0], 'g--', label = "pretrained DeepONet")
      last_idx = FLAGS.tune_steps // FLAGS.tune_record_freq
      plt.plot(x[case_idx,:,-1], tune_pred[case_idx,last_idx,:,0], 'b--', label = f"fine-tuned DeepONet")


      plt.xlabel('$x$')
      plt.ylabel('$c$')
      ymin, ymax = get_ylim([demo_qoi_v[case_idx], qoi[case_idx]])
      plt.ylim(ymin, ymax)
      plt.legend(loc = 'upper right', fontsize = 8)
      plt.title('Quantity of Interest (QoI)')
      
      # plt.suptitle(title)
      plt.tight_layout()
      plt.savefig('{}/{}/{}_case{}.pdf'.format(FLAGS.analysis_folder, FLAGS.icon_stamp, eqn_name, case_idx))
      plt.close('all')

    
def plot_profile_data(eqn_name, result_dict, case_idx_list):
    
    x = result_dict[(eqn_name, 'cond_k')] # (bs, 101, 3), bs = 500
    cond = result_dict[(eqn_name, 'cond_v')] # (bs, 101, 1)
    qoi = result_dict[(eqn_name, 'ground_truth')] # (bs, 101, 1)
    pred = result_dict[(eqn_name, 'pred', 5, -1)] # (bs, 101, 1)
    demo_cond_v = result_dict[(eqn_name, 'demo_cond_v')] # (bs, 5, 101, 1)
    demo_qoi_v = result_dict[(eqn_name, 'demo_qoi_v')] # (bs, 5, 101, 1)
    
    # color_list = ['r', 'g', 'b', 'c', 'm', 'y']
    # default color scheme
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    uxx = laplace_u_batch(cond[...,0], 0.01) # (bs, 101)
    demo_num = demo_cond_v.shape[1]
    for case_idx in case_idx_list:
      plt.figure(figsize=(3.5, 2.5)) # rows, 2 columns
      
      for i in range(demo_num):
        label = "example conditions" if i == 0 else None
        plt.plot(x[case_idx,:,-1], demo_cond_v[case_idx, i,:,0], color = color_list[i], linestyle = '--', alpha = 0.8, label = label)
      plt.plot(x[case_idx,:,-1], cond[case_idx,:,0], 'k--', label = "question condition")

      for i in range(demo_num):
        label = "example QoIs" if i == 0 else None
        plt.plot(x[case_idx,:,-1], demo_qoi_v[case_idx, i,:,0], color = color_list[i], linestyle = '-', alpha = 0.8, label = label)
        
      plt.plot(x[case_idx,:,-1], qoi[case_idx,:,0], 'k-', label = "ground truth QoI")

      plt.xlabel('$x$')
      plt.ylabel('value')
      ymin, ymax = get_ylim([demo_cond_v[case_idx], demo_qoi_v[case_idx],cond[case_idx], qoi[case_idx]])
      plt.ylim(ymin, ymax)
      plt.legend(loc = 'upper right', fontsize = 8)
      
      # plt.suptitle(title)
      plt.tight_layout()
      plt.savefig('{}/{}/data_{}_case{}.pdf'.format(FLAGS.analysis_folder, FLAGS.icon_stamp, eqn_name, case_idx))
      plt.close('all')


def plot_profile_sol(eqn_name, result_dict, pretrain_pred, tune_pred, case_idx_list):
    
    x = result_dict[(eqn_name, 'cond_k')] # (bs, 101, 3), bs = 500
    cond = result_dict[(eqn_name, 'cond_v')] # (bs, 101, 1)
    qoi = result_dict[(eqn_name, 'ground_truth')] # (bs, 101, 1)
    pred = result_dict[(eqn_name, 'pred', 5, -1)] # (bs, 101, 1)
    demo_cond_v = result_dict[(eqn_name, 'demo_cond_v')] # (bs, 5, 101, 1)
    demo_qoi_v = result_dict[(eqn_name, 'demo_qoi_v')] # (bs, 5, 101, 1)
    
    uxx = laplace_u_batch(cond[...,0], 0.01) # (bs, 101)
    demo_num = demo_cond_v.shape[1]
    for case_idx in case_idx_list:
      plt.figure(figsize=(3.5, 2.5)) # rows, 2 columns
      # random opearators
      for t in range(30):
        a = np.random.uniform(0.5, 1.5)
        k = np.random.uniform(0.5, 1.5)
        lamda = 0.1
        c = - lamda * a * uxx[case_idx] + k * cond[case_idx,:,0] ** 3
        label = "random operator" if t == 0 else None
        plt.plot(x[case_idx,:,-1], c, 'k-', alpha = 0.1, label = label)
        
      plt.plot(x[case_idx,:,-1], qoi[case_idx,:,0], 'k-', label = "ground truth")
      plt.plot(x[case_idx,:,-1], pred[case_idx,:,0], 'r--', label = "ICON-LM")

      plt.plot(x[case_idx,:,-1], pretrain_pred[case_idx,:,0], 'g--', label = "pretrained DeepONet")
      last_idx = FLAGS.tune_steps // FLAGS.tune_record_freq
      plt.plot(x[case_idx,:,-1], tune_pred[case_idx,last_idx,:,0], 'b--', label = f"fine-tuned DeepONet")

      plt.xlabel('$x$')
      plt.ylabel('$c$')
      ymin, ymax = get_ylim([pretrain_pred[case_idx,:,0], tune_pred[case_idx,last_idx,:,0], qoi[case_idx]])
      plt.ylim(ymin, ymax)
      plt.legend(loc = 'upper right', fontsize = 8)

      plt.tight_layout()
      plt.savefig('{}/{}/qoi_{}_case{}.pdf'.format(FLAGS.analysis_folder, FLAGS.icon_stamp, eqn_name, case_idx))
      plt.close('all')


def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  eqn_name = 'pde_cubic_spatial_inverse'
  result_dict, deepo_pretrain_pred, deepo_tune_pred, fno_pretrain_pred, fno_tune_pred = get_data(eqn_name)
  if 'error' in FLAGS.mode:
    plot_error(eqn_name, result_dict, deepo_tune_pred, fno_tune_pred)
  if 'sol' in FLAGS.mode:
    plot_profile_sol(eqn_name, result_dict, deepo_pretrain_pred, deepo_tune_pred, list(range(30)))
  if 'data' in FLAGS.mode:
    plot_profile_data(eqn_name, result_dict, list(range(30)))

if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_list('mode', ['error','sol'], 'mode')
  flags.DEFINE_string('analysis_folder', '/home/shared/icon/analysis', 'the folder where analysis results are stored')
  flags.DEFINE_string('icon_stamp', 'icon_lm_learn_20231005-094726-pde3-inverse', 'the stamp of the analysis result')
  flags.DEFINE_string('tune_deepo_stamp', 'icon_lm_deepo_20240121-203825-pde3-inverse', 'the stamp for saving fine-tuning results')
  flags.DEFINE_string('tune_fno_stamp', 'icon_lm_fno_20240121-203841-pde3-inverse', 'the stamp for saving fine-tuning results')

  # select the first tune_steps steps, with tune_record_freq
  flags.DEFINE_integer('tune_record_freq', 10, 'tune record frequency')
  flags.DEFINE_integer('tune_steps', 200, 'tune steps') 
  
  app.run(main)
