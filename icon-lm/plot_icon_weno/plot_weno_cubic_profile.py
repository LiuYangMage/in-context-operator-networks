
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

cmap_1 = 'Blues'
cmap_2 = 'Reds'
# cmap_1 = 'bwr'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import matplotlib.patches as patches

def get_ylim(demo, quest):
  demomin, demomax = np.min(demo), np.max(demo)
  questmin, questmax = np.min(quest), np.max(quest)
  ymin = min(demomin, questmin)
  ymax = max(demomax, questmax)
  gap = (ymax - ymin)
  return ymin - gap * 0.1, ymax + gap * 0.45    

def draw_profile(eqn_name, a, b, c, demo_num, title, case_idx = 0):
    with open("{}/result_dict.pkl".format(folder), "rb") as file:
        result_dict = pickle.load(file)
    with open("{}/consistency_dict.pkl".format(folder), "rb") as file:
        consistency_dict = pickle.load(file)

    caption_id = -1

    x = result_dict[(eqn_name, a, b, c, 'cond_k')] # (bs, 100, 3)
    cond = result_dict[(eqn_name, a, b, c, 'cond_v')] # (bs, 100, 1)
    qoi = result_dict[(eqn_name, a, b, c, 'ground_truth')] # (bs, 100, 1)
    pred = result_dict[(eqn_name, a, b, c, 'pred', demo_num, caption_id)] # (bs, 100, 1)
    demo_cond_v = result_dict[(eqn_name, a, b, c, 'demo_cond_v')] # (bs, 5, 100, 1)
    demo_qoi_v = result_dict[(eqn_name, a, b, c, 'demo_qoi_v')] # (bs, 5, 100, 1)

    print("x.shape", x.shape, "cond.shape", cond.shape, "qoi.shape", qoi.shape, "pred.shape", pred.shape, 
          "demo_cond_v.shape", demo_cond_v.shape, "demo_qoi_v.shape", demo_qoi_v.shape)
    if 'backward' in eqn_name:
      forward = consistency_dict[(eqn_name, a, b, c, 'forward', demo_num, caption_id)] # (bs, 100, 1)


    plt.figure(figsize=(1 * 3.5, 2 * 3)) # 2 rows, 1 columns
    plt.subplot(2, 1, 1) # condition
    for i in range(demo_num):
      plt.plot(x[case_idx,:,-1], demo_cond_v[case_idx, i,:,0], linestyle = ':', alpha = 0.8)
    plt.plot(x[case_idx,:,-1], cond[case_idx,:,0], 'k-', label = "question condition")
    if 'backward' in eqn_name:
      plt.plot(x[case_idx,:,-1], forward[case_idx,:,0], 'b--', label = "forward simulation")
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    ymin, ymax = get_ylim(demo_cond_v[case_idx], cond[case_idx])
    plt.ylim(ymin, ymax)
    plt.legend(loc = 'upper right')
    if 'backward' in eqn_name:
       plt.title('Reverse Condition')
    else:
      plt.title('Forward Condition')

    plt.subplot(2, 1, 2) # QoI
    for i in range(demo_num):
      plt.plot(x[case_idx,:,-1], demo_qoi_v[case_idx, i,:,0], linestyle = ':', alpha = 0.8)
    if 'forward' in eqn_name:
      plt.plot(x[case_idx,:,-1], qoi[case_idx,:,0], 'k-', label = "ground truth")
      plt.plot(x[case_idx,:,-1], pred[case_idx,:,0], 'r--', label = "prediction")
    else:
      plt.plot(x[case_idx,:,-1], pred[case_idx,:,0], 'r-', label = "prediction")
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    ymin, ymax = get_ylim(demo_qoi_v[case_idx], qoi[case_idx])
    plt.ylim(ymin, ymax)
    plt.legend(loc = 'upper right')
    if 'backward' in eqn_name:
       plt.title('Reverse QoI')
    else:
      plt.title('Forward QoI')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('{}/profile_{}_{}_{}_{}_demo{}_case{}.pdf'.format(folder, eqn_name, a, b, c, demo_num, case_idx))
    plt.close('all')

if __name__ == "__main__":
  demo_num_list = [1,2,3,4,5]

  folder = "/home/shared/icon/analysis/icon_weno_20231209-222440_light"
  a,b,c = 0.5,0.5,0.5
  title = "$\partial_t u + \partial_x(0.5u^3 + 0.5u^2 + 0.5u) = 0$"
  for case_idx in range(5):
    draw_profile(eqn_name = 'conservation_weno_cubic_forward', a = a, b = b, c = c, demo_num = 5, title = title, case_idx = case_idx)
    draw_profile(eqn_name = 'conservation_weno_cubic_backward', a = a, b = b, c = c, demo_num = 5, title = title, case_idx = case_idx)

  a,b,c = -0.5,-0.5,-0.5
  title = "$\partial_t u + \partial_x(-0.5u^3 - 0.5u^2 - 0.5u) = 0$"
  for case_idx in range(5):
    draw_profile(eqn_name = 'conservation_weno_cubic_forward', a = a, b = b, c = c, demo_num = 5, title = title, case_idx = case_idx)
    draw_profile(eqn_name = 'conservation_weno_cubic_backward', a = a, b = b, c = c, demo_num = 5, title = title, case_idx = case_idx)
