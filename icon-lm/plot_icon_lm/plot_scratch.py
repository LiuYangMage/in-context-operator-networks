
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

'''
This code will compare fine-tune GPT-2 with training from scratch.
For each model, we will compare training captions v.s. testing captions to see generalization.
'''

def get_join(demo_num_list, folder_dict):

  all_error = {}
  for label, folder_list in folder_dict.items():
    all_error[label] = []

    for folder in folder_list:
      with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
      
        print("{}: {}".format(label, folder))
        plot_key = [i[0] for i in result_dict.keys()] # 19 types, 19 keys
        plot_key = list(OrderedDict.fromkeys(sorted(plot_key))) # remove duplicates and keep order

        relative_error_list = []

        caption_id = 0

        for key in plot_key:
          relative_error_list.append([get_error_from_dict(result_dict, key, demo_num, caption_id)[1] for demo_num in demo_num_list])

        relative_error_list = np.array(relative_error_list)
        relative_error = np.mean(relative_error_list, axis = 0) 
        all_error[label].append(relative_error)

  pprint(plot_key)
  for k, v in all_error.items():
    print(k, v)

  return all_error


def plot_train_vs_test():
  
  pretrained_stamp = "20231014-194955"
  unpretrained_stamp = "20240104-214007"

  cap = 'precise'
  demo_num_list = [0]
  style_dict = {'pretrained GPT-2, training captions': {'line': '--', 'color': 'red', 'marker': 'o'},
                'pretrained GPT-2, testing captions': {'line': '-', 'color': 'red', 'marker': 'o'},
                'unpretrained GPT-2, training captions': {'line': '--', 'color': 'blue', 'marker': 's'},
                'unpretrained GPT-2, testing captions': {'line': '-', 'color': 'blue', 'marker': 's'},
                }
  folder_dict = {
                  'pretrained GPT-2, training captions': ["/home/shared/icon/analysis/icon_gpt2_full_s1-{}-testdata-traincap-{}".format(pretrained_stamp, cap)],
                  'pretrained GPT-2, testing captions': ["/home/shared/icon/analysis/icon_gpt2_full_s1-{}-testdata-testcap-{}".format(pretrained_stamp, cap)],
                  'unpretrained GPT-2, training captions': ["/home/shared/icon/analysis/icon_gpt2_unpretrained-s1-{}-testdata-traincap-{}".format(unpretrained_stamp, cap)],
                  'unpretrained GPT-2, testing captions': ["/home/shared/icon/analysis/icon_gpt2_unpretrained-s1-{}-testdata-testcap-{}".format(unpretrained_stamp, cap)],
                  }
  all_error = get_join(demo_num_list, folder_dict)
  
  '''
  plt.figure(figsize=(3.5, 2.5))
  plt.bar(0, all_error['pretrained GPT-2, training captions'][0], color = 'red', width = 0.2, label = 'pretrained GPT-2, training captions')
  plt.bar(0.2, all_error['pretrained GPT-2, testing captions'][0], color = 'red', width = 0.2, label = 'pretrained GPT-2, testing captions')
  plt.bar(0.4, all_error['unpretrained GPT-2, training captions'][0], color = 'blue', width = 0.2, label = 'unpretrained GPT-2, training captions')
  plt.bar(0.6, all_error['unpretrained GPT-2, testing captions'][0], color = 'blue', width = 0.2, label = 'unpretrained GPT-2, testing captions')
  plt.legend(loc = 'upper left')
  plt.xticks([0, 0.2, 0.4, 0.6], ['pretrained GPT-2', 'pretrained GPT-2', 'unpretrained GPT-2', 'unpretrained GPT-2'])
  plt.ylabel('Relative error')
  plt.ylim([0, 0.05])
  plt.tight_layout()
  plt.savefig('plot_unpretrained_vs_pretrained.pdf')
  '''


def main(argv):
  del argv
  plot_train_vs_test()
  
if __name__ == '__main__':

  FLAGS = flags.FLAGS

  app.run(main)


