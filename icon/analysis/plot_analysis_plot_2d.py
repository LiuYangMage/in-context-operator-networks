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


def myplot():

  analysis_folder = FLAGS.analysis_folder
  stamp = FLAGS.stamp
  dataset_name = FLAGS.dataset_name
  patterns = ['mfc_gparam_hj_forward22']


  k_dim = FLAGS.k_dim
  k_mode = FLAGS.k_mode
  plot_num = FLAGS.plot_num
  demo_num = FLAGS.demo_num
  demo_num_max = 5
  vmin = FLAGS.vmin
  vmax = FLAGS.vmax
  evmin = FLAGS.evmin
  evmax = FLAGS.evmax
  rhocmap = 'Blues'
  errcmap = 'bwr'
  x_size = 100

  with open("{}/err_{}_{}_{}_{}.pickle".format(analysis_folder, stamp, dataset_name, demo_num, demo_num + 1), 'rb') as file:
    results = pickle.load(file)

  for equation in results:
    if not pattern_match(patterns, equation): # key match the patterns
      continue

    this_result = results[equation]
    prompt = this_result['prompt']
    query = this_result['query']
    query_mask = this_result['query_mask']
    pred = this_result['pred']
    gt = this_result['gt']
    cond_k_index, qoi_k_index = get_plot_k_index(k_mode, equation)
    print(equation)
    print(f'cond_k_index: {cond_k_index}, qoi_k_index: {qoi_k_index}')
    for term in this_result.keys(): 
      if hasattr(this_result[term], 'shape'):
        print(term, this_result[term].shape)

    for plot_index in range(0, plot_num * 10, 10):
      # plot quest
      plt.close('all')

      fig = plt.figure(figsize=(9, 3))
      gs = gridspec.GridSpec(1, 7, width_ratios=[0.1, 0.22, 1, 1, 1, 1, 0.1])  # Create a 1x5 grid of subplots

      # Subplot 1: condition
      ax1 = plt.subplot(gs[2])
      cond_k = this_result['raw_quest_cond_k'][plot_index, 0].reshape((-1, x_size, 2)) # [t_size, x_size, 2]
      cond_v = this_result['raw_quest_cond_v'][plot_index, 0].reshape((-1, x_size)) # [t_size, x_size]
      plt.pcolormesh(cond_k[...,0], cond_k[...,1], cond_v, shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
      mask_cond_i = np.abs(prompt[plot_index, :, -1] - 1) < 0.01 # around 1
      plt.plot(prompt[plot_index, mask_cond_i, cond_k_index-1], 
              prompt[plot_index, mask_cond_i, cond_k_index], 's', markersize=3, color = 'black', alpha = 0.7)
      plt.xlabel(r'$t$')
      plt.ylabel(r'$x$')
      plt.title("question\ncondition")

      # Subplot 2: ground truth
      ax2 = plt.subplot(gs[3])
      t, x, v = sort_k_v(query[plot_index], gt[plot_index], k_dim, k_mode, x_size)
      plt.pcolormesh(t, x, v, shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
      plt.xlabel(r'$t$')
      plt.yticks([])
      plt.title("question\n QoI ground truth")

      # Subplot 3: prediction
      ax3 = plt.subplot(gs[4])
      t, x, v = sort_k_v(query[plot_index], pred[plot_index], k_dim, k_mode, x_size)
      plt.pcolormesh(t, x, v, shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
      plt.xlabel(r'$t$')
      plt.yticks([])
      plt.title("question\n QoI prediction")

      # Subplot 4: prediction error
      ax4 = plt.subplot(gs[5])
      t, x, v = sort_k_v(query[plot_index], pred[plot_index] - gt[plot_index], k_dim, k_mode, x_size)
      plt.pcolormesh(t, x, v, shading = shading, vmin = evmin, vmax = evmax, cmap = errcmap)
      plt.xlabel(r'$t$')
      plt.yticks([])
      plt.title("question\n QoI error")

      # Create shared color bar for subplots 1, 2, and 3
      cbar_ax1 = plt.subplot(gs[0])  # Position of the shared color bar
      cbar1 = fig.colorbar(ax1.collections[0], cax=cbar_ax1)
      cbar_ax1.yaxis.set_ticks_position('left') 

      # Create separate color bar for subplot 4
      cbar_ax2 = plt.subplot(gs[6])  # Separate color bar axes
      cbar2 = fig.colorbar(ax4.collections[0], cax=cbar_ax2)

      fig.subplots_adjust(hspace=0.1, wspace=0.1)
      plt.savefig(f"{analysis_folder}/{equation}_index{plot_index}_demo{demo_num}_question.png", dpi = 600, bbox_inches='tight')
      
      # plot demos
      for demo_i in range(demo_num):
        plt.close('all')
        plt.figure(figsize=(3,3))
        cond_k = this_result['raw_demo_cond_k'][plot_index, demo_i] # [cond_tx_size, 2]
        cond_v = this_result['raw_demo_cond_v'][plot_index, demo_i] # [cond_tx_size, 1]
        qoi_k = this_result['raw_demo_qoi_k'][plot_index, demo_i] # [qoi_tx_size, 2]
        qoi_v = this_result['raw_demo_qoi_v'][plot_index, demo_i] # [qoi_tx_size, 1]
        cond_qoi_k = np.concatenate([cond_k, qoi_k], axis=0).reshape((-1, x_size, 2)) # [t_size, x_size, 2]
        cond_qoi_v = np.concatenate([cond_v, qoi_v], axis=0).reshape((-1, x_size)) # [t_size, x_size]
        plt.pcolormesh(cond_qoi_k[...,0], cond_qoi_k[...,1], cond_qoi_v, shading = shading, vmin = vmin, vmax = vmax, cmap = rhocmap)
        
        mask_cond_i = np.abs(prompt[plot_index, :, -demo_num_max-1+demo_i] - 1) < 0.01 # around 1
        plt.plot(prompt[plot_index, mask_cond_i, cond_k_index-1], 
                prompt[plot_index, mask_cond_i, cond_k_index], 'o', markersize=3, color = 'blue', alpha = 0.7)
        
        mask_qoi_i = np.abs(prompt[plot_index, :, -demo_num_max-1+demo_i] + 1) < 0.01 # around -1
        plt.plot(prompt[plot_index, mask_qoi_i, cond_k_index-1], 
                prompt[plot_index, mask_qoi_i, cond_k_index], 'o', markersize=3, color = 'red', alpha = 0.7)
        
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x$')
        plt.title(f"demo #{demo_i + 1}")
        plt.tight_layout()
        plt.savefig(f"{analysis_folder}/{equation}_index{plot_index}_demo{demo_num}_{demo_i}.png", dpi = 600, bbox_inches='tight')


def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  myplot()

if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_integer('k_dim', 3, 'k dimension')
  flags.DEFINE_string('k_mode', 'itx', 'k mode')
  flags.DEFINE_integer('plot_num', 5, 'plot num')
  flags.DEFINE_integer('demo_num', 3, 'demo num, max 5')
  flags.DEFINE_float('vmin', 0.0, 'vmin')
  flags.DEFINE_float('vmax', 2.0, 'vmax')
  flags.DEFINE_float('evmin', -0.04, 'error vmin')
  flags.DEFINE_float('evmax', 0.04, 'error vmax')
  flags.DEFINE_string('analysis_folder', '../analysis/analysis0511a-v4-len-50-50-50-2600', 'the folder where analysis results are stored')
  flags.DEFINE_string('stamp', '20230515-094404_1000000', 'the stamp of the analysis result')
  flags.DEFINE_string('dataset_name', 'data0511a', 'the name of the dataset')

  app.run(main)

