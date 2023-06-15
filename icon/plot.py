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

def plot_all_in_one(equation, prompt, mask, query, query_mask, ground_truth, pred,
                  demo_num, k_dim, v_dim, k_mode,
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
    pred: 2D array, [len(qoi), qoi_v_dim]
    demo_num: int, number of demos
    k_dim: int, max dim for k in prompt
    v_dim: int, max dim for v in prompt
    k_mode: the mode for the key
  @return
    the figure
  '''
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
  fig.suptitle("eqn:{}".format(equation))
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
  
  

if __name__ == "__main__":
  jax.config.update("jax_enable_x64", True)
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  from models import pad_and_concat
  demos = []
  demo_num = 5
  for i in range(demo_num-1):
    demo_cond_k = np.arange(i+1)[:,None] * 0.1 + i
    demo_cond_v = np.arange(i+1)[:,None] * 0.2 + i
    demo_qoi_k = np.arange(i+1)[:,None] * 0.3 + i
    demo_qoi_v = np.arange(i+1)[:,None] * 0.4 + i
    demos.append((demo_cond_k,demo_cond_v, demo_qoi_k,demo_qoi_v))
    print("the {}th demo cond k is {}, cond v is {}, qoi k is {}, qoi v is {}".format(i, demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v))
  quest_cond = (np.arange(6)[:,None]*10, np.arange(6)[:,None] * 10.1)
  prompt, mask = pad_and_concat(demos = demos, quest_cond = quest_cond, k_dim = 3, v_dim = 4, cond_len = 6, qoi_len = 6, demo_num = demo_num)
  print(prompt.shape, mask.shape)
  
  quest_qoi_k = np.arange(7)[:,None] * 20
  quest_qoi_v = np.arange(7)[:,None] * 30
  quest_mask = np.ones((7,))
  quest_qoi_v_true = np.arange(7)[:,None] * 30.1
  fig = plot_all_in_one("test", prompt, mask, quest_qoi_k, quest_mask, quest_qoi_v_true, quest_qoi_v, demo_num = demo_num, k_dim = 3, v_dim = 4, to_tfboard = False)
  filename = "./allfigs_{}.png".format("test")
  fig.savefig(filename)
