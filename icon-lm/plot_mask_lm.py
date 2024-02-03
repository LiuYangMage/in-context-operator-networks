import jax
import jax.numpy as jnp
import numpy as np
import utils
from flax import linen as nn
import transformer_flax as transformer
from transformer_flax import translate_config
from einshape import jax_einshape as einshape
import jax.tree_util as tree
import matplotlib.pyplot as plt


import models_gpt2_icon as models
import dataloader


def plot_mask(mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list, caption_len):
  lw = 4
  lw_b = 12
  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(111)
  ax.imshow(mask * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)
  # add a line to separate caption and the rest, lines are in the boundary of the pixels
  if caption_len > 0:
    ax.axvline(x = caption_len - 0.5, color = 'black', lw = lw, ls = '-')
    ax.axhline(y = caption_len - 0.5, color = 'black', lw = lw, ls = '-')

  # add a line to separate each pair, lines are in the boundary of the pixels
  cursor = caption_len
  for i in range(len(cond_len_list)-1):
    pair_size = cond_len_list[i] + qoi_kv_len_list[i] + qoi_k_len_list[i]
    ax.axvline(x = cursor + pair_size - 0.5, color = 'black', lw = lw, ls = '--')
    ax.axhline(y = cursor + pair_size - 0.5, color = 'black', lw = lw, ls = '--')
    cursor += pair_size

  # add a line to separate cond, qoi_kv, qoi_k, lines are in the boundary of the pixels
  cursor = caption_len
  for i in range(len(cond_len_list)):
    ax.axvline(x = cursor + cond_len_list[i] - 0.5, color = 'black', lw = 1, ls = ':')
    ax.axhline(y = cursor + cond_len_list[i] - 0.5, color = 'black', lw = 1, ls = ':')
    cursor += cond_len_list[i]
    ax.axvline(x = cursor + qoi_kv_len_list[i] - 0.5, color = 'black', lw = 1, ls = ':')
    ax.axhline(y = cursor + qoi_kv_len_list[i] - 0.5, color = 'black', lw = 1, ls = ':')
    cursor += qoi_kv_len_list[i]
    cursor += qoi_k_len_list[i]

  # add the ticks in axis
  ticks = [-0.5, caption_len-0.5]
  for i in range(len(cond_len_list)):
    ticks.append(ticks[-1] + cond_len_list[i])
    ticks.append(ticks[-1] + qoi_kv_len_list[i])
    ticks.append(ticks[-1] + qoi_k_len_list[i])
  ticks = np.array(ticks)
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')

  # draw line segments on y axis
  for pos in [-0.5, mask.shape[0]-0.5]:
    if caption_len > 0:
      plt.plot([pos, pos], [0-0.5, caption_len-0.5], color = '#FF3333', lw = lw_b, ls = '-', solid_capstyle='butt')
    cursor = caption_len
    for i in range(len(cond_len_list)):
      if cond_len_list[i] > 0:
        plt.plot([pos, pos], [cursor-0.5, cursor + cond_len_list[i]-0.5], color = '#72B2F4', lw = lw_b, ls = '-', solid_capstyle='butt'); cursor += cond_len_list[i]
      if qoi_kv_len_list[i] > 0:
        plt.plot([pos, pos], [cursor-0.5, cursor + qoi_kv_len_list[i]-0.5], color = '#75F5B3', lw = lw_b, ls = '-', solid_capstyle='butt'); cursor += qoi_kv_len_list[i]
      if qoi_k_len_list[i] > 0:
        plt.plot([pos, pos], [cursor-0.5, cursor + qoi_k_len_list[i]-0.5], color = '#FF8000', lw = lw_b, ls = '-', solid_capstyle='butt'); cursor += qoi_k_len_list[i]

  # draw line segments on x axis
  for pos in [-0.5, mask.shape[0]-0.5]:
    if caption_len > 0:
      plt.plot([0-0.5, caption_len-0.5], [pos, pos], color = '#FF3333', lw = lw_b, ls = '-', solid_capstyle='butt')
    cursor = caption_len
    for i in range(len(cond_len_list)):
      if cond_len_list[i] > 0:
        plt.plot([cursor-0.5, cursor + cond_len_list[i]-0.5], [pos, pos], color = '#72B2F4', lw = lw_b, ls = '-', solid_capstyle='butt'); cursor += cond_len_list[i]
      if qoi_kv_len_list[i] > 0:
        plt.plot([cursor-0.5, cursor + qoi_kv_len_list[i]-0.5], [pos, pos], color = '#75F5B3', lw = lw_b, ls = '-', solid_capstyle='butt'); cursor += qoi_kv_len_list[i]
      if qoi_k_len_list[i] > 0:
        plt.plot([cursor-0.5, cursor + qoi_k_len_list[i]-0.5], [pos, pos], color = '#FF8000', lw = lw_b, ls = '-', solid_capstyle='butt'); cursor += qoi_k_len_list[i]
  plt.tight_layout()
  return fig



def test():
  from dataloader import DataProvider, print_eqn_caption
  from pprint import pprint
  import jax.tree_util as tree
  np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
  import haiku as hk
  rng = hk.PRNGSequence(42)

  demo_num = 2
  quest_num = 1
  demo_cond_len = 5
  demo_qoi_len = 5
  quest_cond_len = 5
  quest_qoi_len = 5
  k_dim = 3
  v_dim = 1
  caption_len = 10

  data = dataloader.Data(input_id = np.ones((1,1,caption_len)), # (num_devices, batch_size, len)
                        embedding_raw = np.ones((1,1,caption_len,10)), # (num_devices, batch_size, len, embedding_dim) or (num_devices, batch_size, num, len, embedding_dim)
                        embedding_pool = np.ones((1,1,10)), # (num_devices, batch_size, embedding_dim) or (num_devices, batch_size, num, embedding_dim)
                        embedding_mask = np.ones((1,1,caption_len)), # (num_devices, batch_size, len) or (num_devices, batch_size, num, len)
                        demo_cond_k = np.ones((1,1,demo_num,demo_cond_len,k_dim)), # (num_devices, batch_size, demo_num, demo_cond_len, k_dim)
                        demo_cond_v = np.ones((1,1,demo_num,demo_cond_len,v_dim)), # (num_devices, batch_size, demo_num, demo_cond_len, v_dim)
                        demo_cond_mask = np.ones((1,1,demo_num,demo_cond_len)), # (num_devices, batch_size, demo_num, demo_cond_len)
                        demo_qoi_k = np.ones((1,1,demo_num,demo_qoi_len,k_dim)), # (num_devices, batch_size, demo_num, demo_qoi_len, k_dim)
                        demo_qoi_v = np.ones((1,1,demo_num,demo_qoi_len,v_dim)), # (num_devices, batch_size, demo_num, demo_qoi_len, v_dim)
                        demo_qoi_mask = np.ones((1,1,demo_num,demo_qoi_len)), # (num_devices, batch_size, demo_num, demo_qoi_len)
                        quest_cond_k = np.ones((1,1,quest_num,quest_cond_len,k_dim)), # (num_devices, batch_size, quest_num, quest_cond_len, k_dim)
                        quest_cond_v = np.ones((1,1,quest_num,quest_cond_len,v_dim)), # (num_devices, batch_size, quest_num, quest_cond_len, v_dim)
                        quest_cond_mask = np.ones((1,1,quest_num,quest_cond_len)), # (num_devices, batch_size, quest_num, quest_cond_len)
                        quest_qoi_k = np.ones((1,1,quest_num,quest_qoi_len,k_dim)), # (num_devices, batch_size, quest_num, quest_qoi_len, k_dim)
                        quest_qoi_mask = np.ones((1,1,quest_num,quest_qoi_len)), # (num_devices, batch_size, quest_num, quest_qoi_len)
                        )


  data = tree.tree_map(lambda x: x[0,0], data) # take off device and batch dimension
  data_shape = tree.tree_map(lambda x: x.shape, data)

  config = {
  "demo_id_dim": 4,
  "demo_max_num": 6,
  "index_mode": "learn",
  "k_dim": 3,
  "v_dim": 1,
  "causal": "caption",
  "caption_len": 300,
  "input_net": {"hidden_dim": 1024},
  "output_net": {"hidden_dim": 1024}
  }
  # full
  basic_mask, _, _, cond_len_list, qoi_kv_len_list, qoi_k_len_list = models.build_matrices_from_data_shape(data_shape, config = config, compact = True, mode = 'full', 
                                                                                                          caption_len = caption_len, shot_num_min = 0, return_shape_list = True)
  fig = plot_mask(basic_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list, caption_len)
  fig.savefig('mask_icon_gpt_full.pdf')


  # train with caption
  basic_mask, _, _, cond_len_list, qoi_kv_len_list, qoi_k_len_list = models.build_matrices_from_data_shape(data_shape, config = config, compact = True, mode = 'train', 
                                                                                                          caption_len = caption_len, shot_num_min = 0, return_shape_list = True)
  fig = plot_mask(basic_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list, caption_len)
  fig.savefig('mask_icon_gpt_forward_with_caption.pdf')
  
  # train without caption
  basic_mask, _, _, cond_len_list, qoi_kv_len_list, qoi_k_len_list = models.build_matrices_from_data_shape(data_shape, config = config, compact = True, mode = 'train',
                                                                                                          caption_len = 0, shot_num_min = 1, return_shape_list = True)
  fig = plot_mask(basic_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list, 0)
  fig.savefig('mask_icon_gpt_forward_without_caption.pdf')

  # predict with caption
  basic_mask, _, _, cond_len_list, qoi_kv_len_list, qoi_k_len_list = models.build_matrices_from_data_shape(data_shape, config = config, compact = True, mode = 'test',
                                                                                                          caption_len = caption_len, shot_num_min = 0, return_shape_list = True)
  fig = plot_mask(basic_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list, caption_len)
  fig.savefig('mask_icon_gpt_predict_with_caption.pdf')

  # predict without caption
  basic_mask, _, _, cond_len_list, qoi_kv_len_list, qoi_k_len_list = models.build_matrices_from_data_shape(data_shape, config = config, compact = True, mode = 'test',
                                                                                                          caption_len = 0, shot_num_min = 1, return_shape_list = True)
  fig = plot_mask(basic_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list, 0)
  fig.savefig('mask_icon_gpt_predict_without_caption.pdf')


if __name__ == "__main__":
  test()