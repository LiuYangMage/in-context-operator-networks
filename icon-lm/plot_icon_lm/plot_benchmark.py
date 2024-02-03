
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from jax.config import config
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from collections import OrderedDict
from pprint import pprint
import jax.tree_util as tree
from absl import app, flags, logging
from plot_utils import get_error_from_dict

FLAGS = flags.FLAGS


def draw_join(demo_num_list, style_dict, folder_dict, title = None, plot = plt.plot, figsize=(4,3), ylim = (None, None)):

  all_error = {}
  for label, folder_list in folder_dict.items():
    all_error[label] = []

    for folder in folder_list:
      with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
      
        print("{}: {}".format(label, folder))
        plot_key = [i[0] for i in result_dict.keys()]
        plot_key = list(OrderedDict.fromkeys(sorted(plot_key))) # remove duplicates and keep order

        relative_error_list = []

        if ('ode_auto_const_forward', 'error', 1, 0) in result_dict:
          caption_id = 0
        else:
          caption_id = -1

        for key in plot_key:
          relative_error_list.append([get_error_from_dict(result_dict, key, demo_num, caption_id)[1] for demo_num in demo_num_list])

        relative_error_list = np.array(relative_error_list)
        relative_error = np.mean(relative_error_list, axis = 0) 
        all_error[label].append(relative_error)

  pprint(plot_key)
  print(tree.tree_map(lambda x: x.shape, all_error))
  plt.figure(figsize=figsize)
  for label, folder_list in folder_dict.items():
    error_mean = np.mean(all_error[label], axis = 0)
    error_std = np.std(all_error[label], axis = 0)
    plot(demo_num_list, error_mean, label= label, linestyle= style_dict[label]['line'], 
            marker= style_dict[label]['marker'], markersize=7, color= style_dict[label]['color'])
    if len(all_error[label]) > 1:
      plt.fill_between(demo_num_list, error_mean - error_std, error_mean + error_std, alpha=0.2, color= style_dict[label]['color'])

  plt.xticks(demo_num_list)
  plt.xlabel('number of examples')
  plt.ylabel('relative error')
  # ymin = 0.01
  plt.ylim(ylim)
  # ax.set_ylim(figure_config[fi]['ylim'])
  # grid on
  plt.grid(True, which='both', axis='both', linestyle=':')
  plt.legend()
  if title is not None:
    plt.title(title)
    plt.tight_layout()
    plt.savefig('{}/ind_err_join_{}.pdf'.format(folder, title.replace(" ","_")))
    print('saved to {}/ind_err_join_{}.pdf'.format(folder, title.replace(" ","_")))
  else:
    plt.tight_layout()
    plt.savefig('{}/ind_err_join_err.pdf'.format(folder))
    print('saved to {}/ind_err_join_err.pdf'.format(folder))
  
  plt.close('all')


def plot_benchmark():
  '''
  compare ICON-LM with baseline, small models
  '''
  demo_num_list=[1,2,3,4,5]

  style_dict = {
                'ICON-LM (ours)': {'line': '-', 'color': 'red', 'marker': 'o'},
                'Encoder-Decoder ICON': {'line': '--', 'color': 'blue', 'marker': 's'},
                }

  folder_dict = {
                'ICON-LM (ours)': ["/home/shared/icon/analysis/icon_lm_learn_s1-20231005-094726", 
                                  "/home/shared/icon/analysis/icon_lm_learn_s2-20231008-110255",
                                  "/home/shared/icon/analysis/icon_lm_learn_s3-20231009-173624"],
                'Encoder-Decoder ICON': ["/home/shared/icon/analysis/v1baseline_s1-20231004-103259",
                                         "/home/shared/icon/analysis/v1baseline_s2-20231006-114142",
                                         "/home/shared/icon/analysis/v1baseline_s3-20231007-175037"]
                }

  draw_join(demo_num_list, style_dict, folder_dict, figsize=(3.5,2.5))

  

def main(argv):
  del argv
  plot_benchmark()

if __name__ == '__main__':

  FLAGS = flags.FLAGS

  app.run(main)


