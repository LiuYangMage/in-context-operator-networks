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
from pprint import pprint
from dataloader import Data


def get_runner(model_config):

  utils.set_seed(FLAGS.seed)

  opt_config = {'peak_lr': FLAGS.lr,
                  'end_lr': FLAGS.lr,
                  'warmup_steps': 1, # the first step is warmup, i.e. lr = 0
                  'decay_steps': FLAGS.tune_steps,
                  'gnorm_clip': 1,
                  'weight_decay': 0.0001,
                  }

  from runner_deepo_torch import Runner
  runner = Runner(data = None, model_config = model_config, opt_config = opt_config, 
                  model_name = FLAGS.model_name, pretrained = False, 
                  trainable_mode = 'all',
                  loss_mode = ['demo'], # only fine-tune with demo
                  )

  return runner


def tune(eqn_name):

  analysis_folder = FLAGS.analysis_folder
  icon_stamp = FLAGS.icon_stamp
  tune_stamp = FLAGS.tune_stamp

  with open("{}/{}/result_dict.pkl".format(analysis_folder, icon_stamp), 'rb') as file:
    result_dict = pickle.load(file)
  
  model_config = utils.load_json('../config_model/{}'.format(FLAGS.model_config))
  runner = get_runner(model_config=model_config)

  data_template = Data(input_id = np.zeros((0,)),
                embedding_raw = np.zeros((0,)), 
                embedding_pool = np.zeros((0,)), 
                embedding_mask = np.zeros((0,)), 
                demo_cond_k = np.zeros((0,)), 
                demo_cond_v = np.zeros((0,)), 
                demo_cond_mask = np.zeros((0,)), 
                demo_qoi_k = np.zeros((0,)), 
                demo_qoi_v = np.zeros((0,)), 
                demo_qoi_mask = np.zeros((0,)), 
                quest_cond_k = np.zeros((0,)), 
                quest_cond_v = np.zeros((0,)), 
                quest_cond_mask = np.zeros((0,)), 
                quest_qoi_k = np.zeros((0,)), 
                quest_qoi_mask = np.zeros((0,)), 
                )
  
  if 'notune' in FLAGS.mode:
    runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
  
    this_data = data_template._replace(quest_cond_k = result_dict[(eqn_name, 'cond_k')][:,None,:,:],
                                      quest_cond_v = result_dict[(eqn_name, 'cond_v')][:,None,:,:], 
                                      quest_qoi_k = result_dict[(eqn_name, 'qoi_k')][:,None,:,:])
    deepo_error, deepo_pred = runner.get_error(this_data, result_dict[(eqn_name, 'ground_truth')][:,None,:,:], 
                                               with_caption = False, return_pred = True) # (bs, query_len, 1)
    
    print("pred", deepo_pred.shape, np.mean(deepo_pred), np.std(deepo_pred)) # (bs, query_len, 1)
    print("error", deepo_error.shape, np.mean(deepo_error), np.std(deepo_error)) # (bs)

    deepo_results_dict = result_dict.copy()
    deepo_results_dict[(eqn_name, 'pred', 5, -1)] = deepo_pred
    deepo_results_dict[(eqn_name, 'error', 5, -1)] = deepo_error

    if not os.path.exists("{}/{}/".format(analysis_folder, tune_stamp)):
      os.mkdir("{}/{}/".format(analysis_folder, tune_stamp))
      
    with open("{}/{}/deepo_pretrain_result_dict.pkl".format(analysis_folder, tune_stamp), 'wb') as file:
      pickle.dump(deepo_results_dict, file)

  if 'tune' in FLAGS.mode:
    loss_results = []
    pred_results = []
    error_results = []
    for bid in range(int(FLAGS.tune_bid_range[0]), int(FLAGS.tune_bid_range[1])):
      runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
      runner.reset_optimizer()
      this_data = data_template._replace(demo_cond_k = result_dict[(eqn_name, 'demo_cond_k')][bid:bid+1],
                                        demo_cond_v = result_dict[(eqn_name, 'demo_cond_v')][bid:bid+1],
                                        demo_qoi_k = result_dict[(eqn_name, 'demo_qoi_k')][bid:bid+1],
                                        demo_qoi_v = result_dict[(eqn_name, 'demo_qoi_v')][bid:bid+1],
                                        quest_cond_k = result_dict[(eqn_name, 'cond_k')][bid:bid+1,None,:,:],
                                        quest_cond_v = result_dict[(eqn_name, 'cond_v')][bid:bid+1,None,:,:],
                                        quest_qoi_k = result_dict[(eqn_name, 'qoi_k')][bid:bid+1,None,:,:]
                                        )
      
      init_deepo_error, init_deepo_pred = runner.get_error(this_data, result_dict[(eqn_name, 'ground_truth')][bid:bid+1,None,:,:], 
                                                           with_caption = False, return_pred = True) # (bs, query_len, 1)
      this_loss = []
      this_pred = []
      this_error = []
      for n in range(runner.opt_config['decay_steps']+1):
        runner.iter(this_data, 0) # should not use label
        if n % FLAGS.tune_record_freq == 0:
          print('bid', bid, 'n', n, end = " ")
          loss = runner.get_loss(this_data, 0)
          print("loss", loss, end = " ")
          deepo_error, deepo_pred = runner.get_error(this_data, result_dict[(eqn_name, 'ground_truth')][bid:bid+1,None,:,:], 
                                                    with_caption = False, return_pred = True) # (bs, query_len, 1)
          print("error", np.mean(deepo_error), np.mean(init_deepo_error), flush=True)
          this_loss.append(loss)
          this_pred.append(deepo_pred)
          this_error.append(deepo_error)
      loss_results.append(this_loss)
      pred_results.append(this_pred)
      error_results.append(this_error)
    loss_results = np.array(loss_results) # (bs, N)
    pred_results = np.array(pred_results) # (bs, N, 1, query_len, 1)
    error_results = np.array(error_results) # (bs, N, 1)

    pred_results = pred_results[:,:,0,:,:] # (bs, N, query_len, 1)
    error_results = error_results[:,:,0] # (bs, N)

    print("loss", loss_results.shape, np.mean(loss_results, axis = 0))
    print("pred", pred_results.shape)
    print("error", error_results.shape, np.mean(error_results, axis = 0), flush=True)

    deepo_tune_results_dict = result_dict.copy()
    deepo_tune_results_dict[(eqn_name, 'loss', 5, -1)] = loss_results
    deepo_tune_results_dict[(eqn_name, 'pred', 5, -1)] = pred_results
    deepo_tune_results_dict[(eqn_name, 'error', 5, -1)] = error_results

    if not os.path.exists("{}/{}/".format(analysis_folder, tune_stamp)):
      os.mkdir("{}/{}/".format(analysis_folder, tune_stamp))

    with open("{}/{}/deepo_tune_result_dict_{}_{}.pkl".format(
              analysis_folder, tune_stamp, int(FLAGS.tune_bid_range[0]), int(FLAGS.tune_bid_range[1])), 'wb') as file:
      pickle.dump(deepo_tune_results_dict, file)

    
def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  tune(eqn_name = 'pde_cubic_spatial_inverse')

if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_list('mode', ['notune', 'tune'], 'mode')
  flags.DEFINE_integer('seed', 1, 'seed')
  flags.DEFINE_string('analysis_folder', '/home/shared/icon/analysis', 'the folder where analysis results are stored')
  flags.DEFINE_string('icon_stamp', 'icon_lm_learn_20231005-094726-pde3-inverse', 'the stamp of the data for fine-tuning')
  flags.DEFINE_string('tune_stamp', 'icon_lm_deepo_20240120-015041-pde3-inverse', 'the stamp for saving fine-tuning results')
  flags.DEFINE_string('restore_dir', '/home/shared/icon/save/user/ckpts/deepo_pretrain/20240120-015041', 
                      'the folder where ckpt results are stored')
  flags.DEFINE_string('model_name', 'deepo', 'model name')
  flags.DEFINE_string('model_config', 'model_deepo_pde_config.json', 'model config')
  flags.DEFINE_integer('restore_step', 100000, 'the step of the ckpt to restore')
  flags.DEFINE_integer('tune_steps', 1000, 'tune steps')
  flags.DEFINE_integer('tune_record_freq', 10, 'tune record frequency')
  flags.DEFINE_list('tune_bid_range', [0,500], 'range of tune batch id')
  flags.DEFINE_float('lr', 0.00001, 'learning rate')


  app.run(main)
