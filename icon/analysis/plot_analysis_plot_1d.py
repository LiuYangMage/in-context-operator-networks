import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from plot_analysis_utils import label_map

def pattern_match(patterns, name):
  for pattern in patterns:
    if pattern in name:
      return True
  return False


def myplot():

  analysis_folder = FLAGS.analysis_folder
  stamp = FLAGS.stamp
  dataset_name = FLAGS.dataset_name
  patterns = ['ode', 'series', 'pde', 'mfc']


  k_dim = FLAGS.k_dim
  k_mode = FLAGS.k_mode
  plot_num = FLAGS.plot_num
  demo_num = FLAGS.demo_num
  demo_num_max = 5

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
    cond_k_index, qoi_k_index = get_plot_k_index(k_mode, equation)
    raw_cond_k_index, raw_qoi_k_index = get_plot_k_index('naive', equation)

    print(equation)
    print(f'cond_k_index: {cond_k_index}, qoi_k_index: {qoi_k_index}')
    print(f'raw_cond_k_index: {raw_cond_k_index}, raw_qoi_k_index: {raw_qoi_k_index}')
    for term in ["raw_demo_cond_k","raw_demo_cond_v", "raw_demo_qoi_k", "raw_demo_qoi_v",
                  "raw_quest_cond_k","raw_quest_cond_v","raw_quest_qoi_k","raw_quest_qoi_v"]:
      print(term, this_result[term].shape)


    for plot_index in range(0, plot_num * 10, 10):
      plt.close('all')
      plt.figure(figsize=(4,5))
      plt.subplot(2, 1, 1)  
      # plot demo conditions
      for demo_i in range(demo_num):
        plt.plot(this_result['raw_demo_cond_k'][plot_index, demo_i, :, raw_cond_k_index], 
                this_result['raw_demo_cond_v'][plot_index, demo_i, :, 0], ':', alpha = 0.7) 
        mask_cond_i = np.abs(prompt[plot_index, :, -demo_num_max-1+demo_i] - 1) < 0.01 # around 1
        plt.plot(prompt[plot_index, mask_cond_i, cond_k_index], 
                prompt[plot_index, mask_cond_i, k_dim], 'o', markersize=3, color = 'grey', alpha = 0.7)
      # plot quest conditions
      plt.plot(this_result['raw_quest_cond_k'][plot_index, 0, :, raw_cond_k_index], 
              this_result['raw_quest_cond_v'][plot_index, 0, :, 0], 'k-') 
      mask_cond_i = np.abs(prompt[plot_index, :, -1] - 1) < 0.01 # around 1
      plt.plot(prompt[plot_index, mask_cond_i, cond_k_index], 
              prompt[plot_index, mask_cond_i, k_dim], 's', markersize=3, color = 'blue', alpha = 0.7)
      plt.title(label_map[equation]['legend']+'\ncondition')
      plt.xlabel(label_map[equation]['xlabel'])

      # plot demo qois
      plt.subplot(2, 1, 2)
      for demo_i in range(demo_num):
        plt.plot(this_result['raw_demo_qoi_k'][plot_index, demo_i, :, raw_qoi_k_index], 
                this_result['raw_demo_qoi_v'][plot_index, demo_i, :, 0], ':', alpha = 0.7) 
        mask_qoi_i = np.abs(prompt[plot_index, :, -demo_num_max-1+demo_i] + 1) < 0.01 # around -1
        plt.plot(prompt[plot_index, mask_qoi_i, qoi_k_index], 
                prompt[plot_index, mask_qoi_i, k_dim], 'o', markersize=3, color = 'grey', alpha = 0.7)
      # plot quest qois, i.e., ground truth and predictions
      plt.plot(this_result['raw_quest_qoi_k'][plot_index, 0, :, raw_qoi_k_index], 
              this_result['raw_quest_qoi_v'][plot_index, 0, :, 0], 'k-')
      plt.plot(query[plot_index, query_mask[plot_index], qoi_k_index], 
              pred[plot_index, query_mask[plot_index], 0], 's', markersize=3, color = 'red', alpha = 0.7)
      plt.title('QoI')
      plt.xlabel(label_map[equation]['xlabel'])


      plt.tight_layout()
      plt.savefig(f"{analysis_folder}/{equation}_demo{demo_num}_index{plot_index}.pdf",
              format = 'pdf', bbox_inches='tight')
      plt.title(this_result['equation'][plot_index]) # for debugging
      plt.savefig(f"{analysis_folder}/{equation}_demo{demo_num}_index{plot_index}_equation.pdf",
              format = 'pdf', bbox_inches='tight')


def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  myplot()

if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_integer('k_dim', 3, 'k dimension')
  flags.DEFINE_string('k_mode', 'itx', 'k mode')
  flags.DEFINE_integer('plot_num', 5, 'plot num for each equation, should correspond to different operators')
  flags.DEFINE_integer('demo_num', 5, 'demo num used, max 5')

  flags.DEFINE_string('analysis_folder', '../analysis/analysis0511a-v4-ind', 'the folder where analysis results are stored')
  flags.DEFINE_string('stamp', '20230515-094404_1000000', 'the stamp of the analysis result')
  flags.DEFINE_string('dataset_name', 'data0511a', 'the name of the dataset')

  app.run(main)
