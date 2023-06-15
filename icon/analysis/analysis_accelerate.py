from jax.config import config
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from pprint import pprint
import sys
sys.path.append('../')

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from functools import partial
import haiku as hk
import optax
import pytz
from datetime import datetime
import pickle
from absl import app, flags, logging
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from einshape import jax_einshape as einshape
from utils import load_json
import gc

import utils
import glob
from pprint import pprint
import dataloader
import plot
import run

def get_key(task, eqn_name):
  '''
  get the key for the eqn_name
  @param 
    task: string, the task name, 'ind', 'ood'
    eqn_name: string, the name of the equation
  @return key: tuple
  '''
  if task == 'ind' or task == 'len':
    eqn_name_split = eqn_name.split("_")
    eqn_name_clean = "_".join(eqn_name_split[:4])
    key = eqn_name_clean
  elif task == 'ood':
    if "ode_auto_const" in eqn_name:
      coeffs = eqn_name.split("_")
      # amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
      # bmins = np.linspace(-3, 3, FLAGS.ood_coeff2_grids)[:-1]; gap_b = bmins[1] - bmins[0]
      # all_params.append("{:.3f}_{:.3f}".format(coeff_a, coeff_b))
      coeff1_buck = int(np.floor(float(coeffs[4])* 10))/10 # buck size 0.1
      coeff2_buck = int(np.floor(float(coeffs[5])* 5))/5 # buck size 0.2
      key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
    elif "ode_auto_linear1" in eqn_name:  # e.g., ode_auto_linear1_forward_0.200_-0.091
      coeffs = eqn_name.split("_")
      # amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
      # bmins = np.linspace(-3, 3, FLAGS.ood_coeff2_grids)[:-1]; gap_b = bmins[1] - bmins[0]
      # all_params.append("{:.3f}_{:.3f}".format(coeff_a, coeff_b))
      coeff1_buck = int(np.floor(float(coeffs[4])* 10))/10 # buck size 0.1
      coeff2_buck = int(np.floor(float(coeffs[5])* 5))/5 # buck size 0.2
      key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
    elif "ode_auto_linear2" in eqn_name:  # e.g., ode_auto_linear2_forward_0.128_0.556_1.156
      coeffs = eqn_name.split("_")
      # amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
      # bmins = np.linspace(-3, 3, FLAGS.ood_coeff2_grids)[:-1]; gap_b = bmins[1] - bmins[0]
      # all_params.append("{:.3f}_{:.3f}_{:.3f}".format(coeff_a1, coeff_a, coeff_b))
      coeff1_buck = int(np.floor(float(coeffs[5])* 10))/10 # buck size 0.1
      coeff2_buck = int(np.floor(float(coeffs[6])* 5))/5 # buck size 0.2
      key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
    elif "pde_porous_spatial" in eqn_name:  # e.g., pde_porous_spatial_forward_0.128_0.560_-5.248_0.404, four num: ul, ur, c, a
      coeffs = eqn_name.split("_")
      # amins = np.linspace(0.1, 4, FLAGS.ood_coeff1_grids)[:-1]; gap_a = amins[1] - amins[0]
      # cmins = np.linspace(-6, 6, FLAGS.ood_coeff2_grids)[:-1]; gap_c = cmins[1] - cmins[0]
      # all_params.append("{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(coeff_ul, coeff_ur, coeff_c, coeff_a))
      coeff1_buck = int(np.floor(float(coeffs[7])* 10))/10  # coeffa, buck size 0.1
      coeff2_buck = int(np.floor(float(coeffs[6])* 2.5))/2.5 # coeffc, buck size 0.4
      key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff2_buck)
    elif "series_damped_oscillator" in eqn_name:  # e.g., series_damped_oscillator_forward_-0.624
      coeffs = eqn_name.split("_")
      # decays_grids = np.linspace(-1.0, 5.0, FLAGS.ood_coeff1_grids)[:-1]; gap = decays_grids[1] - decays_grids[0]
      # all_params.append("{:.3f}".format(decay))
      coeff1_buck = int(np.floor(float(coeffs[4])* 6.66))/6.66  # decay, buck size 0.15
      key = ("_".join(eqn_name.split("_")[:4]), coeff1_buck, coeff1_buck)
    elif "ode_auto_linear3" in eqn_name:
      coeff = float(eqn_name.split("_")[5]) # use the second coeff as key
      key =  ("_".join(eqn_name.split("_")[:4]), coeff)
    elif "mfc" in eqn_name:
      eqn_name_split = eqn_name.split("_")
      eqn_name_clean = "_".join(eqn_name_split[:4])
      key = eqn_name_clean
    else:
      raise NotImplementedError
  return key


def apply_operator(equation_list, prompt, fake = True):
  '''du/dt = a1 * c * u + a2 * u + a3'''
  pred = []
  for i, this_eqn in enumerate(equation_list):
    this_a, this_b, this_c = [float(j) for j in this_eqn.split("_")[4:7]]
    this_b = 0.0 if fake else this_b
    this_cond = prompt[i,500:, FLAGS.k_dim] # [50], value of the condition
    cond_len = int(np.sum(prompt[i, 500:, -1])) # int, length of the condition with mask = True
    if 'inverse' in this_eqn:
      this_u = this_cond[:cond_len-1]
      next_u = this_cond[1:cond_len]
      rhs = (next_u - this_u)/0.02
      c = (rhs -  this_c - this_u * this_b)/(this_u * this_a)
      c = jnp.pad(c, (0, 50-len(c)), constant_values = 0.0) # pad to 50
      pred.append(c)
    elif 'forward' in this_eqn:
      u_init = this_cond[-1]
      control = this_cond[:cond_len-1]
      u = [u_init]
      for j in range(cond_len-1):
        rhs = this_a * control[j] * u[-1] + this_b * u[-1] + this_c
        u.append(u[-1] + 0.02 * rhs)
      u = jnp.array(u)
      u = jnp.pad(u, (0, 50-len(u)), constant_values = 0.0) # pad to 50
      pred.append(u)
    else:
      raise NotImplementedError
  return jnp.stack(pred)[...,None] # [bs, 50, 1]


def clean_data(prompt, mask, query, query_mask, ground_truth, list_size = 0):
    if list_size > 0:
      prompt = jnp.concatenate(prompt)
      mask = jnp.concatenate(mask)
      query = jnp.concatenate(query)
      query_mask = jnp.concatenate(query_mask)
      ground_truth = jnp.concatenate(ground_truth)
    prompt = einshape('ij...->(ij)...', prompt)
    mask = einshape('ij...->(ij)...', mask)
    query = einshape('ij...->(ij)...', query)
    query_mask = einshape('ij...->(ij)...', query_mask)
    ground_truth = einshape('ij...->(ij)...', ground_truth)
    return prompt, mask, query, query_mask, ground_truth


def is_fake_demo_consistency(equation, fake_equation):
  for eqn, fake_eqn in zip(equation, fake_equation):
    eqn = eqn.split("_")
    fake_eqn = fake_eqn.split("_")
    if eqn[-1] != fake_eqn[-1]:
      return False
    if eqn[-3] != fake_eqn[-3]:
      return False
    if float(fake_eqn[-2])!= 0.0:
      return False
  return True


def rebuild_prompt(equation, prompt, mask, query, query_mask, ground_truth):
  '''
  assume all data have been selected, 
  now only select the first part to feed into the model
  '''
  assert np.shape(prompt)[-2] == (FLAGS.cond_len + FLAGS.qoi_len) * FLAGS.demo_num + FLAGS.cond_len
  assert  np.shape(mask)[-1] == (FLAGS.cond_len + FLAGS.qoi_len) * FLAGS.demo_num + FLAGS.cond_len
  new_prompt = []; new_mask = []
  start = 0
  for i in range(FLAGS.demo_num):
    end = start + FLAGS.cond_len
    new_prompt.append(prompt[..., start:end, :][..., :FLAGS.len_demo_cond_len, :])
    new_mask.append(mask[..., start:end][..., :FLAGS.len_demo_cond_len])
    start = end; end = start + FLAGS.qoi_len
    new_prompt.append(prompt[..., start:end, :][..., :FLAGS.len_demo_qoi_len, :])
    new_mask.append(mask[..., start:end][..., :FLAGS.len_demo_qoi_len])
    start = end
  end = start + FLAGS.cond_len
  new_prompt.append(prompt[..., start:end, :][...,:FLAGS.len_quest_cond_len, :])
  new_mask.append(mask[..., start:end][...,:FLAGS.len_quest_cond_len])
  assert end == np.shape(prompt)[-2]
  prompt = np.concatenate(new_prompt, axis = -2)
  mask = np.concatenate(new_mask, axis = -1)

  query = query[...,:FLAGS.len_quest_qoi_len, :]
  query_mask = query_mask[...,:FLAGS.len_quest_qoi_len]
  ground_truth = ground_truth[...,:FLAGS.len_quest_qoi_len, :]

  return equation, prompt, mask, query, query_mask, ground_truth


def write_errdict(errdict, equation, prompt, mask, query, query_mask, ground_truth, pred, raw):
  error = np.linalg.norm(ground_truth - pred, axis = -1)
  for eqni, eqn_name in enumerate(equation):
    key = get_key(FLAGS.task, eqn_name)
    if key not in errdict:
      errdict[key] = {"pred": [], "gt": [], "error": [], "query_mask": [],
                      "equation": [], "prompt": [], "mask": [], "query": [],
                      "raw_demo_cond_k" : [], "raw_demo_cond_v" : [], "raw_demo_qoi_k" : [], "raw_demo_qoi_v" : [],
                      "raw_quest_cond_k" : [], "raw_quest_cond_v" : [], "raw_quest_qoi_k" : [], "raw_quest_qoi_v" : []}
    errdict[key]["pred"].append(pred[eqni,...])
    errdict[key]["gt"].append(ground_truth[eqni,...])
    errdict[key]["error"].append(error[eqni,...])
    errdict[key]["query_mask"].append(query_mask[eqni,...])
    if FLAGS.save_raw:
      errdict[key]["raw_demo_cond_k"].append(raw[0][eqni,...])
      errdict[key]["raw_demo_cond_v"].append(raw[1][eqni,...])
      errdict[key]["raw_demo_qoi_k"].append(raw[2][eqni,...])
      errdict[key]["raw_demo_qoi_v"].append(raw[3][eqni,...])
      errdict[key]["raw_quest_cond_k"].append(raw[4][eqni,...])
      errdict[key]["raw_quest_cond_v"].append(raw[5][eqni,...])
      errdict[key]["raw_quest_qoi_k"].append(raw[6][eqni,...])
      errdict[key]["raw_quest_qoi_v"].append(raw[7][eqni,...])
    if FLAGS.save_prompt:
      errdict[key]["equation"].append(eqn_name)
      errdict[key]["prompt"].append(prompt[eqni,...])
      errdict[key]["mask"].append(mask[eqni,...])
      errdict[key]["query"].append(query[eqni,...])


def plot_figure(figdict, equation, prompt, mask, query, query_mask, ground_truth, pred):
  for fi in range(len(equation)):
    key = get_key('ind', equation[fi])
    if key not in figdict:
      figdict[key] = 0
    if figdict[key] < FLAGS.figs:
      figdict[key] += 1
      figure = plot.plot_all_in_one(equation[fi], prompt[fi], mask[fi], query[fi], query_mask[fi], ground_truth[fi], pred[fi],
                              demo_num= FLAGS.demo_num, k_dim = FLAGS.k_dim, v_dim = FLAGS.v_dim, k_mode = FLAGS.k_mode, to_tfboard=False) 
      
      eqn_str = get_key('ind', equation[fi])
      folder = f"{FLAGS.analysis_dir}/{FLAGS.task}_{eqn_str}"
      if not os.path.exists(folder):
        os.makedirs(folder)
      filename = f"{folder}/{FLAGS.stamp}_fig{figdict[key]}_{equation[fi]}_{FLAGS.demo_num_begin}_{FLAGS.demo_num_end}.pdf"
      figure.savefig(filename, format='pdf', bbox_inches='tight')
      plt.close('all')


@utils.timeit
def get_errdict(data, runner, errdict, figdict, fake_demo_data = None):
  '''
  load all data, use runner to get prediction, and save the results to errdict
  '''
  read_step = 0
  while True:
    utils.print_dot(read_step)
    read_step += 1
    try:
      raw, equation, prompt, mask, query, query_mask, ground_truth = data.get_next_data(decode_equation = True, return_raw = True)
    except StopIteration:
      break
    
    if fake_demo_data is not None:
      fake_equation, fake_prompt, fake_mask, _, _, _ = fake_demo_data.get_next_data(decode_equation = True, return_raw = False)
      # the order of k-v pairs may vary, but the equation/operator should be consistent
      assert is_fake_demo_consistency(equation, fake_equation)
      demo_len = (FLAGS.cond_len + FLAGS.qoi_len) * FLAGS.demo_num
      prompt = jnp.concatenate([fake_prompt[...,:demo_len,:], prompt[...,demo_len:,:]], axis = -2)
      mask = jnp.concatenate([fake_mask[...,:demo_len], mask[...,demo_len:]], axis = -1)
    
    if FLAGS.task == 'len':
      equation, prompt, mask, query, query_mask, ground_truth = rebuild_prompt(equation, prompt, mask, query, query_mask, ground_truth)
      if read_step == 1:
        print("prompt: ", np.shape(prompt), "sum: ", np.sum(np.abs(prompt[...,(FLAGS.k_dim + FLAGS.v_dim):])), flush=True)
        print("mask: ", np.shape(mask), "sum: ", np.sum(mask), flush=True)
        print("query: ", np.shape(query), flush=True)
        print("query_mask: ", np.shape(query_mask), "sum: ", np.sum(query_mask), flush=True)
        print("ground_truth: ", np.shape(ground_truth), flush=True)
    
    if FLAGS.mode == 'fake_op':
      prompt, mask, query, query_mask, ground_truth = clean_data(prompt, mask, query, query_mask, ground_truth)
      pred = apply_operator(equation, prompt, fake=True)
    elif FLAGS.mode == 'real_op':
      prompt, mask, query, query_mask, ground_truth = clean_data(prompt, mask, query, query_mask, ground_truth)
      pred = apply_operator(equation, prompt, fake=False)
    elif FLAGS.mode in ['vanilla', 'fake_demo']:
      pred = runner.get_pred(prompt, mask, query, use_list = False)
      pred = einshape('ij...->(ij)...', pred)
      prompt, mask, query, query_mask, ground_truth = clean_data(prompt, mask, query, query_mask, ground_truth)
    else:
      raise NotImplementedError
    
    write_errdict(errdict, equation, prompt, mask, query, query_mask, ground_truth, pred, raw)
    plot_figure(figdict, equation, prompt, mask, query, query_mask, ground_truth, pred)

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  if not os.path.exists(FLAGS.analysis_dir):
    os.makedirs(FLAGS.analysis_dir)

  problem = FLAGS.problem
  stamp = FLAGS.stamp
  step = FLAGS.step
  
  k_dim = FLAGS.k_dim
  v_dim = FLAGS.v_dim
  k_mode = FLAGS.k_mode

  demo_num = FLAGS.demo_num
  cond_len = FLAGS.cond_len
  qoi_len = FLAGS.qoi_len
  batch_size = FLAGS.batch_size
  shuffle_buffer_size = FLAGS.shuffle_buffer_size

  seed = FLAGS.seed
  prompt_dim = k_dim + v_dim + demo_num + 1
  query_dim = k_dim
  qoi_v_dim = v_dim
  hidden_dim = FLAGS.hidden_dim
  num_heads = FLAGS.num_heads
  num_layers = FLAGS.num_layers
  optimizer = optax.adamw(0.0001) # dummy optimizer
  
  test_config = load_json(FLAGS.test_config_filename)

  for key, value in test_config.items():
    if 'demo_num_begin' in value:
      value['demo_num_begin'] = FLAGS.demo_num_begin
    if 'demo_num_end' in value:
      value['demo_num_end'] = FLAGS.demo_num_end

  print("test_config: ", flush=True)
  pprint(test_config)
  save_dir = "../check_points/{}/{}".format(problem, stamp)

  runner = run.Runner(seed = seed, 
                      prompt_dim = prompt_dim,
                      query_dim = query_dim, 
                      qoi_v_dim = qoi_v_dim,
                      hidden_dim = hidden_dim,
                      num_heads = num_heads,
                      num_layers = num_layers,
                      optimizer = optimizer,
                  )

  runner.restore(save_dir = save_dir, step = step, restore_opt_state = False)

  test_data_dirs = ["../data_generation/{}".format(i) for i in FLAGS.test_data_dirs]
  test_files = []
  for test_data_dir in test_data_dirs:
    print("test_data_dir: ", test_data_dir, flush=True)
    for test_data_glob in FLAGS.test_data_globs:
      path = os.path.join(test_data_dir, test_data_glob)
      files = glob.glob(path)
      test_files.extend(files)
  test_files.sort()
  # test_files = [i for i in test_files if '_inverse.' in i] # filter
  # test_files = test_files[:10]
  print('=========test files========')
  pprint(test_files)
  print('===========test files length {}========='.format(len(test_files)), flush=True)

  errdict = {}
  figdict = {}
  if FLAGS.mode in ['fake_demo']:
    for test_file in test_files:
      if 'ode_auto_linear3_forward.tfrecord' in test_file:
        demo_file = [i for i in test_files if  "_0.00_" in i and "ode_auto_linear3_forward.tfrecord" in i][0]
      elif 'ode_auto_linear3_inverse.tfrecord' in test_file:
        demo_file = [i for i in test_files if  "_0.00_" in i and "ode_auto_linear3_inverse.tfrecord" in i][0]
      else:
        raise NotImplementedError
      print("test_file: ", test_file, flush=True)
      print("demo_file: ", demo_file, flush=True)
      data = dataloader.DataProvider(seed = seed,
                          demo_num = demo_num,
                          cond_len = cond_len,
                          qoi_len = qoi_len,
                          batch_size = batch_size,
                          shuffle_buffer_size = shuffle_buffer_size, 
                          file_names = test_file,
                          k_dim = k_dim, v_dim = v_dim, 
                          config = test_config,
                          select = 'ordered',
                          k_mode = k_mode,
                          deterministic = True,
                          return_raw = FLAGS.save_raw,
                          drop_remainder = False,
                          shuffle_dataset = False,
                          num_epochs=1)
      fake_demo_data = dataloader.DataProvider(seed = seed,
                          demo_num = demo_num,
                          cond_len = cond_len,
                          qoi_len = qoi_len,
                          batch_size = batch_size,
                          shuffle_buffer_size = shuffle_buffer_size, 
                          file_names = demo_file,
                          k_dim = k_dim, v_dim = v_dim, 
                          config = test_config,
                          select = 'ordered',
                          k_mode = k_mode,
                          deterministic = True,
                          return_raw = FLAGS.save_raw,
                          drop_remainder = False,
                          shuffle_dataset = False,
                          num_epochs=1)
      get_errdict(data, runner, errdict, figdict, fake_demo_data)
      gc.collect()

  elif FLAGS.mode in ['vanilla', 'fake_op', 'real_op']:
    data = dataloader.DataProvider(seed = seed,
                        demo_num = demo_num,
                        cond_len = cond_len,
                        qoi_len = qoi_len,
                        batch_size = batch_size,
                        shuffle_buffer_size = shuffle_buffer_size, 
                        file_names = test_files,
                        k_dim = k_dim, v_dim = v_dim, 
                        config = test_config,
                        select = 'ordered',
                        k_mode = k_mode,
                        deterministic = True,
                        return_raw = FLAGS.save_raw,
                        drop_remainder = False,
                        shuffle_dataset = False,
                        num_epochs=1)
    get_errdict(data, runner, errdict, figdict)
  else:
    raise NotImplementedError
  
  for key, value in errdict.items(): # key is the equation name
    value["pred"] = np.stack(value["pred"], axis = 0)
    value["gt"] = np.stack(value["gt"], axis = 0)
    value["error"] = np.stack(value["error"], axis = 0)
    value["query_mask"] = np.stack(value["query_mask"], axis = 0).astype(bool)
    if FLAGS.save_raw:
      value["raw_demo_cond_k"] = np.stack(value["raw_demo_cond_k"], axis = 0)
      value["raw_demo_cond_v"] = np.stack(value["raw_demo_cond_v"], axis = 0)
      value["raw_demo_qoi_k"] = np.stack(value["raw_demo_qoi_k"], axis = 0)
      value["raw_demo_qoi_v"] = np.stack(value["raw_demo_qoi_v"], axis = 0)
      value["raw_quest_cond_k"] = np.stack(value["raw_quest_cond_k"], axis = 0)
      value["raw_quest_cond_v"] = np.stack(value["raw_quest_cond_v"], axis = 0)
      value["raw_quest_qoi_k"] = np.stack(value["raw_quest_qoi_k"], axis = 0)
      value["raw_quest_qoi_v"] = np.stack(value["raw_quest_qoi_v"], axis = 0)
    if FLAGS.save_prompt:
      value['prompt'] = np.stack(value['prompt'], axis = 0)
      value['mask'] = np.stack(value['mask'], axis = 0)
      value['query'] = np.stack(value['query'], axis = 0)
    error = value["error"]
    query_mask = value["query_mask"]
    ground_truth = value["gt"]
    error_mean = np.mean(error, where=query_mask)
    error_std = np.std(error, where=query_mask)
    gtmean = np.mean(np.linalg.norm(ground_truth, axis = -1), where=query_mask)
    value['error_mean'] = error_mean
    value['error_std'] = error_std
    value['gtmean'] = gtmean
    value['relative_error_mean'] = error_mean/gtmean
    value['relative_error_std'] = error_std/gtmean

    print("{} size {} & {:.5f}$\pm${:.5f} & {:.5f}$\pm${:.5f} \\\\".format(
            key, {k: value[k].shape for k in value.keys() if hasattr(value[k], 'shape')},
            error_mean, 2 * error_std,
            error_mean/gtmean, 2 * error_std/gtmean), flush=True)
  
  
  print(f'===================errdict size {len(errdict)}====================', flush=True)
  test_data_dirs_str = "_".join(FLAGS.test_data_dirs)
  if FLAGS.mode in ['vanilla']:
    pkfilename = f"{FLAGS.analysis_dir}/err_{stamp}_{step}_{test_data_dirs_str}_{FLAGS.demo_num_begin}_{FLAGS.demo_num_end}.pickle"
  elif FLAGS.mode in ['fake_demo']:
    pkfilename = f"{FLAGS.analysis_dir}/err_{stamp}_{step}_fake_demo_{test_data_dirs_str}_{FLAGS.demo_num_begin}_{FLAGS.demo_num_end}.pickle"
  elif FLAGS.mode in ['fake_op', 'real_op']:
    pkfilename = f"{FLAGS.analysis_dir}/err_{FLAGS.mode}_{test_data_dirs_str}_{FLAGS.demo_num_begin}_{FLAGS.demo_num_end}.pickle"
  else:
    raise NotImplementedError
  pickle.dump(errdict, open(pkfilename, "wb"))
  print("errdict saved to {}".format(pkfilename))



if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_enum('task', 'ind', ['ind', 'ood', 'len'], 'task type')
  flags.DEFINE_enum('mode', 'vanilla', ['vanilla', 'fake_demo', 'fake_op', 'real_op'], 'mode of analysis')
  flags.DEFINE_string('analysis_dir', 'analysis', 'write file to dir')

  flags.DEFINE_string('problem', 'hero', 'problem')
  flags.DEFINE_string('stamp', '20230511-215453', 'stamp')
  flags.DEFINE_integer('step', 1000000, 'step to restore')
  flags.DEFINE_integer('seed', 1, 'random seed for data generation')

  flags.DEFINE_integer('demo_num_begin', 1, 'demo num begin')
  flags.DEFINE_integer('demo_num_end', 6, 'demo num end')

  flags.DEFINE_list('test_data_dirs', ['data0409a'], 'directories of testing data')
  flags.DEFINE_list('test_data_globs', ['test*'], 'filename glob patterns of testing data')
  flags.DEFINE_string('test_config_filename', 'test_config.json', 'config file for testing')

  flags.DEFINE_integer('demo_num', 5, 'the number of demos')
  flags.DEFINE_integer('cond_len', 50, 'the length of the condition')
  flags.DEFINE_integer('qoi_len', 50, 'the length of the qoi')

  flags.DEFINE_integer('len_demo_cond_len', 100, 'the length of the demo condition for task len')
  flags.DEFINE_integer('len_demo_qoi_len', 100, 'the length of the demo qoi for task len')
  flags.DEFINE_integer('len_quest_cond_len', 100, 'the length of the quest condition for task len')
  flags.DEFINE_integer('len_quest_qoi_len', 2600, 'the length of the quest qoi for task len')

  flags.DEFINE_integer('k_dim', 2, 'the dim of the key')
  flags.DEFINE_integer('v_dim', 1, 'the dim of the value')
  flags.DEFINE_string('k_mode', 'naive', 'mode for keys') 
  flags.DEFINE_integer('qoi_v_dim', 1, 'the output dim of the model, also the dim of qoi_v ')

  # save_raw and save_prompt consume a lot of memory and storage, be careful
  flags.DEFINE_bool('save_raw', False, 'return raw data, be careful, see dataloader.py for details')
  flags.DEFINE_bool('save_prompt', False, 'save equation names, prompts, masks and queries')
  flags.DEFINE_integer('figs', 0, 'num of figures to plot for each key')

  flags.DEFINE_integer('hidden_dim', 256, 'the dim of the model')
  flags.DEFINE_integer('num_heads', 8, 'the number of heads in the model')
  flags.DEFINE_integer('num_layers', 6, 'the number of layers in the model')

  flags.DEFINE_integer('batch_size', 200, 'batch size')
  flags.DEFINE_integer('list_size', 0, 'number of lists')
  flags.DEFINE_integer('shuffle_buffer_size', 1000, 'shuffle buffer size')

  app.run(main)
