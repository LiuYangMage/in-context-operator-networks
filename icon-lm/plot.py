import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import utils
import tensorflow as tf
import os

def plot_pred(equation, prompt, mask, query, ground_truth, pred, to_tfboard = True):
  '''
  plot the figure
  @param 
    equation: string
    prompt: 2D array, [len(prompt), prompt_dim]
    mask: 1D array, [len(prompt)]
    query: 2D array, [len(qoi), qoi_k_dim]
    ground_truth: 2D array, [len(qoi), qoi_v_dim]
    pred: 2D array, [len(qoi), qoi_v_dim]
  @return
    the figure
  '''
  figure = plt.figure(figsize=(6, 4))
  plt.subplot(1,1,1)
  plt.plot(query[:,0], ground_truth[:,0], 'k-', label='Ground Truth')
  plt.plot(query[:,0], pred[:,0], 'r--', label='Prediction')
  plt.xlabel('key')
  plt.ylabel('value')
  plt.title("eqn:{}, qoi".format(equation))
  plt.legend()
  plt.tight_layout()
  if to_tfboard:
    return utils.plot_to_image(figure)
  else:
    return figure

def plot_subfig(ax, t, x_true, x_pred = None, mask = None):
  '''
  plot the subfigure (only plot the first dimension)
  @param 
    ax: the subfigure
    t: 2D array, [len(t), k_dim]
    x_pred: 1D array, [len(t), v_dim]
    x_true: 1D array, [len(t), v_dim]
  @return
    the subfigure
  '''
  if mask is not None:
    t = t[mask]
    x_true = x_true[mask]
    if x_pred is not None:
      x_pred = x_pred[mask]
  ax.plot(t[:,0], x_true[:,0], 'ko', markersize=3, label='GT')
  if x_pred is not None:
    ax.plot(t[:,0], x_pred[:,0], 'ro', markersize=1, label='Pred')
    ax.legend()
  ax.set_xlabel('key')
  ax.set_ylabel('value')
  

def plot_all(equation, prompt, mask, query, query_mask, ground_truth, pred = None,
            demo_num = 5, k_dim = 1, v_dim = 1,
            to_tfboard = True, ):
  '''
  plot all figures in demo and prediction
  @param 
    equation: string
    prompt: 2D array, [len(prompt), prompt_dim]
    mask: 1D array, [len(prompt)]
    query: 2D array, [len(qoi), query_dim]
    query_mask: 1D array, [len(qoi)]
    ground_truth: 2D array, [len(qoi), qoi_v_dim]
    pred: 2D array, [len(qoi), qoi_v_dim], skip if None
    demo_num: int, number of demos
    k_dim: int, max dim for k in prompt
    v_dim: int, max dim for v in prompt
  @return
    the figure
  '''
  fig_col_half_num = 2
  fig_row_num = demo_num // fig_col_half_num + 1
  fig, axs = plt.subplots(fig_row_num, 2* fig_col_half_num, figsize=(12, 8))
  fig.subplots_adjust(hspace=0.5, wspace=0.5)
  fig.suptitle("eqn:{}".format(equation))
  _, prompt_dim = jnp.shape(prompt)
  if prompt_dim != k_dim + v_dim + demo_num + 1:
    raise ValueError("Error in plot: prompt_dim != k_dim + v_dim + demo_num + 1")
  for i in range(demo_num):
    row_ind = i // fig_col_half_num
    col_ind = i % fig_col_half_num
    # plot demo cond
    mask_cond = jnp.abs(prompt[:, -demo_num - 1 + i] - 1) < 0.01 # around 1
    demo_cond_k = prompt[mask_cond, :k_dim]
    demo_cond_v = prompt[mask_cond, k_dim:k_dim+v_dim]
    ax = axs[row_ind, 2*col_ind]
    plot_subfig(ax, demo_cond_k, demo_cond_v)
    ax.set_title("demo {}, cond".format(i))
    # plot demo qoi
    mask_qoi = jnp.abs(prompt[:, -demo_num - 1 + i] + 1) < 0.01 # around -1
    demo_qoi_k = prompt[mask_qoi, :k_dim]
    demo_qoi_v = prompt[mask_qoi, k_dim:k_dim+v_dim]
    ax = axs[row_ind, 2*col_ind+1]
    plot_subfig(ax, demo_qoi_k, demo_qoi_v)
    ax.set_title("demo {}, qoi".format(i))
  # plot pred
  mask_cond = jnp.abs(prompt[:, -1] - 1) < 0.01 # around 1
  cond_k = prompt[mask_cond, :k_dim]
  cond_v = prompt[mask_cond, k_dim:k_dim+v_dim]
  row_ind = demo_num // fig_col_half_num
  col_ind = demo_num % fig_col_half_num
  ax = axs[row_ind, 2*col_ind]
  plot_subfig(ax, cond_k, cond_v)
  ax.set_title("quest, cond")
  ax = axs[row_ind, 2*col_ind+1]
  plot_subfig(ax, query, ground_truth, pred, query_mask.astype(bool))
  ax.set_title("quest, qoi")
  if to_tfboard:
    return utils.plot_to_image(fig)
  else:  # save to a file
    return fig
  

def get_plot_k_index(k_mode, equation):
  if k_mode == 'naive':
    if "gparam" in equation:
      cond_k_index = 1
      qoi_k_index = 1
    elif "rhoparam" in equation:
      cond_k_index = 0
      qoi_k_index = 1
    else:
      cond_k_index = 0
      qoi_k_index = 0
  elif k_mode == 'itx':
    if ("ode" in equation) or ("series" in equation):
      cond_k_index = 1
      qoi_k_index = 1
    else:
      cond_k_index = 2
      qoi_k_index = 2
  else:
    raise NotImplementedError
  return cond_k_index, qoi_k_index

def plot_all_in_one(equation, caption, prompt, mask, query, query_mask, ground_truth, pred,
                  config, to_tfboard = True, ):
  '''
  plot all figures in demo and prediction
  @param 
    equation: string
    prompt: 2D array, [len(prompt), prompt_dim]
    mask: 1D array, [len(prompt)]
    query: 2D array, [len(qoi), query_dim]
    query_mask: 1D array, [len(qoi)]
    ground_truth: 2D array, [len(qoi), qoi_v_dim]
    pred: 2D array, [len(qoi), qoi_v_dim]
    demo_num: int, number of demos
    k_dim: int, max dim for k in prompt
    v_dim: int, max dim for v in prompt
    k_mode: the mode for the key
  @return
    the figure
  '''
  plt.close('all')
  demo_num, k_dim, v_dim, k_mode, = config['demo_num'], config['k_dim'], config['v_dim'], config['k_mode']
  cond_k_index, qoi_k_index = get_plot_k_index(k_mode, equation)
  prompt = np.array(prompt)
  mask = np.array(mask)
  query = np.array(query)
  query_mask = np.array(query_mask)
  ground_truth = np.array(ground_truth)
  pred = np.array(pred)

  # check consistency between mask and prompt
  assert np.sum(mask) == np.sum(np.abs(prompt[:, (k_dim + v_dim):]))

  fig, axs = plt.subplots(3, 1, figsize=(10, 8))
  fig.subplots_adjust(hspace=0.5, wspace=0.0)
  fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
  # plot cond quest
  mask_cond_quest = np.abs(prompt[:, -1] - 1) < 0.01 # around 1
  # cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
  axs[0].plot(prompt[mask_cond_quest, cond_k_index], prompt[mask_cond_quest,k_dim], 'k+', markersize=7, label='cond quest')
  # plot pred
  query_mask = query_mask.astype(bool)
  axs[1].plot(query[query_mask,qoi_k_index], ground_truth[query_mask,0], 'k+', markersize=7, label='ground truth')
  axs[2].plot(query[query_mask,qoi_k_index], ground_truth[query_mask,0], 'k+', markersize=7, label='ground truth')
  axs[2].plot(query[query_mask,qoi_k_index], pred[query_mask,0], 'r+', markersize=7, label='pred')
  cond_mask_nonzero_num = []
  qoi_mask_nonzero_num = []
  for i in range(demo_num):
    mask_cond_i = np.abs(prompt[:, -demo_num-1+i] - 1) < 0.01 # around 1
    mask_qoi_i = np.abs(prompt[:, -demo_num-1+i] + 1) < 0.01 # around -1
    if np.sum(mask_cond_i) > 0 and np.sum(mask_qoi_i) > 0:  # demo that is used
      cond_mask_nonzero_num.append(np.sum(mask_cond_i))
      qoi_mask_nonzero_num.append(np.sum(mask_qoi_i))
      # NOTE: we don't need mask because prompt is multiplied by mask when constructed
      # cond_i = prompt[mask_cond_i, :k_dim+v_dim] # [cond_len_in_use, k_dim+v_dim]
      # qoi_i = prompt[mask_qoi_i, :k_dim+v_dim] # [qoi_len_in_use, k_dim+v_dim]
      axs[0].plot(prompt[mask_cond_i,cond_k_index], prompt[mask_cond_i,k_dim], 'o', markersize=3, label='cond {}'.format(i), alpha = 0.5)
      axs[1].plot(prompt[mask_qoi_i,qoi_k_index], prompt[mask_qoi_i,k_dim], 'o', markersize=3, label='qoi {}'.format(i), alpha = 0.5)
  cond_mask_nonzero_num = np.array(cond_mask_nonzero_num)
  qoi_mask_nonzero_num = np.array(qoi_mask_nonzero_num)
  axs[0].set_xlabel('key'); axs[0].set_ylabel('value')
  axs[1].set_xlabel('key'); axs[1].set_ylabel('value')
  axs[2].set_xlabel('key'); axs[2].set_ylabel('value')
  axs[0].set_title("cond, {} demos mask nonzero num: {}, quest mask nonzero num: {}".format(
                  cond_mask_nonzero_num.shape[0], cond_mask_nonzero_num, np.sum(mask_cond_quest)))
  axs[1].set_title("demo qoi, {} demos mask nonzero num: {}".format(qoi_mask_nonzero_num.shape[0], qoi_mask_nonzero_num))
  axs[2].set_title("quest qoi, mask nonzero num: {}".format(np.sum(query_mask)))
  if to_tfboard:
    return utils.plot_to_image(fig)
  else:  # save to a file
    return fig
  
  

def plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
  '''
  plot all figures in demo and prediction
  @param 
    equation: string
    caption: string
  @return
    the figure
  '''
  plt.close('all')
  cond_k_index, qoi_k_index = get_plot_k_index(config['k_mode'], equation)
  
  fig, axs = plt.subplots(3, 1, figsize=(10, 8))
  fig.subplots_adjust(hspace=0.5, wspace=0.0)
  caption = ""
  fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
  # plot cond quest
  
  # cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
  for i in range(len(data.demo_cond_mask)):
    axs[0].plot(data.demo_cond_k[i, data.demo_cond_mask[i,:].astype(bool), cond_k_index], 
                data.demo_cond_v[i, data.demo_cond_mask[i,:].astype(bool), 0], 'o', markersize=3, label='cond {}'.format(i), alpha = 0.5)
    axs[1].plot(data.demo_qoi_k[i, data.demo_qoi_mask[i,:].astype(bool), qoi_k_index],
                data.demo_qoi_v[i, data.demo_qoi_mask[i,:].astype(bool), 0], 'o', markersize=3, label='qoi {}'.format(i), alpha = 0.5)
  axs[0].plot(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), cond_k_index], 
              data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='cond quest')
  axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), qoi_k_index],
              label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='qoi quest')
  # plot pred
  axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), qoi_k_index],
              label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='qoi quest')
  axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), qoi_k_index],
              pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'r+', markersize=7, label='qoi pred')

  demo_cond_len = np.sum(data.demo_cond_mask, axis=1)
  demo_qoi_len = np.sum(data.demo_qoi_mask, axis=1)
  quest_cond_len = np.sum(data.quest_cond_mask, axis=1)
  quest_qoi_len = np.sum(data.quest_qoi_mask, axis=1)

  axs[0].set_xlabel('key'); axs[0].set_ylabel('value')
  axs[1].set_xlabel('key'); axs[1].set_ylabel('value')
  axs[2].set_xlabel('key'); axs[2].set_ylabel('value')
  axs[0].set_title("cond, demo len: {}, quest len: {}".format(demo_cond_len, quest_cond_len))
  axs[1].set_title("qoi, demo len: {}, quest len: {}".format(demo_qoi_len, quest_qoi_len))
  axs[2].set_title("quest qoi, len: {}".format(quest_qoi_len))
  if to_tfboard:
    return utils.plot_to_image(fig)
  else:  # save to a file
    return fig
  
  

if __name__ == "__main__":
  pass
