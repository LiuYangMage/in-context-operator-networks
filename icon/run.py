from jax.config import config
import tensorflow as tf
import os
from utils import load_json
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from pprint import pprint

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from functools import partial
import haiku as hk
import optax
import pytz
from datetime import datetime
import utils
from absl import app, flags, logging
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import models
from dataloader import DataProvider
import plot
from einshape import jax_einshape as einshape



gpus = tf.config.list_physical_devices(device_type = 'GPU')
print(gpus, flush=True)
print(jax.devices(), flush=True)

class Runner():
  def __init__(self, seed, prompt_dim, query_dim, qoi_v_dim,
               hidden_dim, num_heads, num_layers, 
               optimizer, initializer = 'glorot_uniform',
               devices = jax.devices()):
    
    self.seed = seed
    self.prompt_dim = prompt_dim
    self.query_dim = query_dim
    self.qoi_v_dim = qoi_v_dim

    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.initializer = initializer

    self.devices = devices
    self.num_devices = len(devices)

    self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    self.params, self.predict_fn, self.loss_fn = self.build_basic_fn() # no batch
    self.opt_state = optimizer.init(self.params) # intialize the optimizer
    utils.print_pytree(self.params) # print the params

    # no multi-gpu yet
    self.predict_batch_fn = jax.jit(jax.vmap(self.predict_fn, in_axes = [None, None, 0, 0, 0], out_axes = 0)) # predict in batch  
    self.loss_batch_fn = jax.jit(jax.vmap(self.loss_fn, in_axes = [None, None, 0, 0, 0, 0, 0], out_axes = 0)) # no average over batch
    self.loss_batch_ave_fn = jax.jit(lambda *args, **kwargs: jnp.mean(self.loss_batch_fn(*args, **kwargs))) # average over batch
    
    # now work on the multi-gpu part
    self.params = jax.device_put_replicated(self.params, devices) # replicate the params to all devices
    self.opt_state = jax.device_put_replicated(self.opt_state, devices) # replicate the opt_state to all devices
    self.predict_pmap_batch_fn = jax.pmap(self.predict_batch_fn, axis_name='devices') # make predictions in batch and devices
    self.loss_pmap_batch_fn = jax.pmap(self.loss_batch_fn, axis_name='devices') # no average over batch and devices
    self.loss_pmap_batch_ave_fn = jax.pmap(self.loss_batch_ave_fn, axis_name='devices') # average over batch, no average over devices
    self.train_iter = utils.get_train_iter_pmap(self.loss_batch_ave_fn, optimizer)
    
    self.train_step = 0
  
  def next_key(self):
    '''duplicate the key for each device'''
    return einshape('i->ji', next(self.rng), j = self.num_devices)

  def build_basic_fn(self):
    def f(prompt, mask, query):
      net = models.SolverModel(q_size = self.hidden_dim,
                              kv_size = self.hidden_dim,
                              qoi_v_size = self.qoi_v_dim,
                              QK_size = self.hidden_dim,
                              V_size = self.hidden_dim,
                              num_heads = self.num_heads,
                              num_layers = self.num_layers,
                              initializer = self.initializer,
                              )
      return net(prompt, mask, query)

    f = hk.transform(f)
    prompt = jnp.ones((10, self.prompt_dim))
    mask = jnp.ones((10,))
    query = jnp.ones((5, self.query_dim))
    params = f.init(next(self.rng), prompt, mask, query)

    @jax.jit
    def predict_fn(params, rng_key, prompt, mask, query):
      '''the neural network function without batch'''
      return f.apply(params, rng_key, prompt, mask, query)

    @jax.jit
    def loss_fn(params, rng_key, prompt, mask, query, query_mask, ground_truth):
      '''the loss function without batch'''
      out = predict_fn(params, rng_key, prompt, mask, query)
      loss = jnp.mean((out - ground_truth)**2, where = query_mask[...,None])
      return loss

    return params, predict_fn, loss_fn

  def save(self, save_dir):
    params_path = save_dir + '/{}_params.pickle'.format(self.train_step)
    opt_state_path = save_dir + '/{}_opt_state.pickle'.format(self.train_step)

    # only take the first device
    with open(params_path, 'wb') as file:
      pickle.dump(jax.device_get(jax.tree_map(lambda x: x[0], self.params)), file)
    with open(opt_state_path, 'wb') as file:
      pickle.dump(jax.device_get(jax.tree_map(lambda x: x[0], self.opt_state)), file)
    
    logging.info('saved to {}, step {}'.format(save_dir, self.train_step))

  def restore(self, save_dir, step, restore_opt_state = True):
    self.train_step = step
    params_path = save_dir + '/{}_params.pickle'.format(self.train_step)
    opt_state_path = save_dir + '/{}_opt_state.pickle'.format(self.train_step)

    with open(params_path, 'rb') as file:
      params = pickle.load(file)
      self.params = jax.device_put_replicated(params, self.devices) # replicate the params to all devices
    logging.info('restored params from {}, step {}'.format(save_dir, step))

    if restore_opt_state:
      with open(opt_state_path, 'rb') as file:
        opt_state = pickle.load(file)
        self.opt_state = jax.device_put_replicated(opt_state, self.devices) # replicate the opt_state to all devices
      logging.info('restored opt state from {}, step {}'.format(save_dir, step))
  
  def iter(self, prompt, mask, query, query_mask, ground_truth, use_list = False):
    '''prompt, mask, query, query_mask, ground_truth are of size (num_devices, batch_on_each_device, ...)
    lists of size (num_devices, batch_on_each_device, ...), if use_list is True'''
    train_iter = self.train_iter_list if use_list else self.train_iter # useless
    self.params, self.opt_state = train_iter(self.params, self.next_key(), self.opt_state, prompt, mask, query, query_mask, ground_truth)
    self.train_step += 1

  def get_loss(self, prompt, mask, query, query_mask, ground_truth, use_list = False):
    '''prompt, mask, query, query_mask, ground_truth are of size (num_devices, batch_on_each_device, ...)
    lists of size (num_devices, batch_on_each_device, ...), if use_list is True
    return losses, no average over devices or batch'''
    if use_list:
      losses = []
      for prompt_i, mask_i, query_i, query_mask_i, ground_truth_i in zip(prompt, mask, query, query_mask, ground_truth):
        losses.append(self.loss_pmap_batch_fn(self.params, self.next_key(), prompt_i, mask_i, query_i, query_mask_i, ground_truth_i))
      losses = jnp.concatenate(losses, axis = 0) # (num_devices * list_size, batch_on_each_device)
    else:
      # (num_devices, batch_on_each_device)
      losses = self.loss_pmap_batch_fn(self.params, self.next_key(), prompt, mask, query, query_mask, ground_truth) 
    return losses
  
  def get_pred(self, prompt, mask, query, use_list = False):
    '''prompt, mask, query, are of size (num_devices, batch_on_each_device, ...)
    lists of size (num_devices, batch_on_each_device, ...), if use_list is True
    return predictions'''
    if use_list:
      pred = []
      for prompt_i, mask_i, query_i, in zip(prompt, mask, query):
        pred.append(self.predict_pmap_batch_fn(self.params, self.next_key(), prompt_i, mask_i, query_i))
      pred = jnp.concatenate(pred, axis = 0)
    else:
      # (num_devices, batch_size, query_length, value_dim)
      pred = self.predict_pmap_batch_fn(self.params, self.next_key(), prompt, mask, query)
    return pred


def run_train():
  time_stamp = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S")
  stamp = time_stamp
  print("stamp: {}".format(stamp))

  train_warmup_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_warmup_percent // 100
  train_decay_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_decay_percent // 100

  print("train_decay_steps = {}".format(train_decay_steps), flush=True)
  print("train_warmup_steps = {}".format(train_warmup_steps), flush=True)

  train_data_dirs = FLAGS.train_data_dirs
  if FLAGS.test_data_dirs is None:
    test_data_dirs = FLAGS.train_data_dirs
  else:
    test_data_dirs = FLAGS.test_data_dirs

  train_file_names = ["{}/{}".format(i, j) for i in train_data_dirs for j in FLAGS.train_data_globs]
  test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in FLAGS.test_data_globs]

  print("train_file_names: ", flush=True)
  pprint(train_file_names)
  print("test_file_names: ", flush=True)
  pprint(test_file_names)

  train_config = load_json(FLAGS.train_config_filename)
  if FLAGS.test_config_filename is None:
    test_config = train_config
  else:
    test_config = load_json(FLAGS.test_config_filename)

  print("train_config: ", flush=True)
  pprint(train_config)
  print("test_config: ", flush=True)
  pprint(test_config)
  
  if FLAGS.plot_num is None:
    plot_num = FLAGS.train_batch_size
  else:
    plot_num = FLAGS.plot_num
  
  optimizer = utils.get_scheduled_adamw(peak_lr = FLAGS.train_peak_lr, 
                                        end_lr = FLAGS.train_end_lr,
                                        warmup_steps = train_warmup_steps,
                                        decay_steps = train_decay_steps,
                                        gnorm_clip = FLAGS.train_gnorm_clip,
                                        weight_decay = FLAGS.train_weight_decay)
  train_data = DataProvider(seed = FLAGS.seed + 1,
                            demo_num = FLAGS.demo_num,
                            cond_len = FLAGS.cond_len,
                            qoi_len = FLAGS.qoi_len,
                            batch_size = FLAGS.train_batch_size,
                            shuffle_buffer_size = FLAGS.train_shuffle_buffer_size, 
                            file_names = train_file_names,
                            k_dim = FLAGS.k_dim,
                            v_dim = FLAGS.v_dim,
                            config = train_config,
                            select = 'random',
                            k_mode = FLAGS.k_mode,
                            deterministic = FLAGS.deterministic,
                          )
  test_data = DataProvider(seed = FLAGS.seed + 10,
                            demo_num = FLAGS.demo_num,
                            cond_len = FLAGS.cond_len,
                            qoi_len = FLAGS.qoi_len,
                            batch_size = FLAGS.train_batch_size,
                            shuffle_buffer_size = FLAGS.train_shuffle_buffer_size, 
                            file_names = test_file_names,
                            k_dim = FLAGS.k_dim,
                            v_dim = FLAGS.v_dim,
                            config = test_config,
                            select = 'random',
                            k_mode = FLAGS.k_mode,
                            deterministic = FLAGS.deterministic,
                          )
  exm_equation, exm_prompt, exm_mask, exm_query, exam_query_mask, exm_ground_truth = \
            train_data.get_next_data(decode_equation = True, list_size = 0)
  train_data.pretty_print(exm_equation, exm_prompt, exm_mask, exm_query, exam_query_mask, exm_ground_truth)

  runner = Runner(seed = FLAGS.seed,
                  prompt_dim = exm_prompt.shape[-1],
                  query_dim = exm_query.shape[-1],
                  qoi_v_dim = FLAGS.qoi_v_dim,
                  hidden_dim = FLAGS.hidden_dim, 
                  num_heads = FLAGS.num_heads,
                  num_layers = FLAGS.num_layers,
                  optimizer = optimizer,
                  initializer = FLAGS.initializer,
                  )

  if FLAGS.tfboard:
    results_dir = './results/{}/'.format(FLAGS.problem)+ stamp
    file_writer = tf.summary.create_file_writer(results_dir)
    ckpt_dir = './check_points/{}/'.format(FLAGS.problem)+ stamp
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

  utils.timer.tic("since last print")
  for _ in range(FLAGS.epochs * FLAGS.steps_per_epoch + 1):
    if runner.train_step % (FLAGS.steps_per_epoch * 10) == 0 and FLAGS.tfboard:
      logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))
      runner.save(ckpt_dir)
    
    if runner.train_step % 100 == 0: # print and write loss to tfboard
      utils.timer.toc("since last print")
      utils.timer.tic("since last print")
      utils.timer.tic("print and write loss to tfboard")
      _, prompt, mask, query, query_mask, ground_truth = train_data.get_next_data(list_size = FLAGS.list_size)
      this_train_loss = runner.get_loss(prompt, mask, query, query_mask, ground_truth, use_list= bool(FLAGS.list_size))
      this_train_loss_mean = jnp.mean(this_train_loss); this_train_loss_std = jnp.std(this_train_loss)

      equation, prompt, mask, query, query_mask, ground_truth = test_data.get_next_data(decode_equation = True, list_size = FLAGS.list_size)
      this_test_loss = runner.get_loss(prompt, mask, query, query_mask, ground_truth, use_list= bool(FLAGS.list_size))
      this_test_loss_mean = jnp.mean(this_test_loss); this_test_loss_std = jnp.std(this_test_loss)

      print("step: {}, train loss: {:.4f}+-{:.4f}, test loss: {:.4f}+-{:.4f}".format(
              runner.train_step, this_train_loss_mean, this_train_loss_std, this_test_loss_mean, this_test_loss_std), flush=True)
      print(equation[0:5], flush = True)

      if FLAGS.tfboard:
        with file_writer.as_default():
          tf.summary.scalar('loss/train_loss', this_train_loss_mean, step = runner.train_step)
          tf.summary.scalar('loss/test_loss', this_test_loss_mean, step = runner.train_step)
      utils.timer.toc("print and write loss to tfboard")

    if runner.train_step % 1000 == 0: # plot to tfboard
      utils.timer.tic("plot to tfboard")
      equation, prompt, mask, query, query_mask, ground_truth = test_data.get_next_data(decode_equation = True, list_size = FLAGS.list_size)
      pred = runner.get_pred(prompt, mask, query, use_list = bool(FLAGS.list_size)) # (num_devices * list_size, batch_on_each_device, ...)
      if FLAGS.list_size:
        prompt = jnp.concatenate(prompt)
        mask = jnp.concatenate(mask)
        query = jnp.concatenate(query)
        query_mask = jnp.concatenate(query_mask)
        ground_truth = jnp.concatenate(ground_truth)
      if FLAGS.tfboard:
        with file_writer.as_default():
          for fij in range(plot_num):
              fi = fij // (FLAGS.train_batch_size // runner.num_devices)
              fj = fij % (FLAGS.train_batch_size // runner.num_devices)
              figure = plot.plot_all_in_one(equation[fij], prompt[fi,fj], mask[fi,fj], query[fi,fj], query_mask[fi,fj], ground_truth[fi,fj], pred[fi,fj],
                                    demo_num=FLAGS.demo_num, k_dim = FLAGS.k_dim, v_dim = FLAGS.v_dim, k_mode = FLAGS.k_mode)
              tf.summary.image("test case {}-{}".format(fi, fj), figure, step = runner.train_step)
      utils.timer.toc("plot to tfboard")

    _, prompt, mask, query, query_mask, ground_truth = train_data.get_next_data(list_size=FLAGS.list_size)
    runner.iter(prompt, mask, query, query_mask, ground_truth, use_list= bool(FLAGS.list_size))

    if runner.train_step == 5: # exclude warming up steps
      utils.timer.tic("time estimate")
    if runner.train_step > 0 and runner.train_step % 500 == 0:
      ratio = runner.train_step/(FLAGS.epochs * FLAGS.steps_per_epoch)
      utils.timer.estimate_time("time estimate", ratio)

def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  tf.random.set_seed(FLAGS.seed + 123456) 
  if FLAGS.main == 'train':
    time_stamp = run_train()
  elif FLAGS.main == 'test':
    raise NotImplementedError


if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_boolean('tfboard', False, 'dump into tfboard')
  flags.DEFINE_boolean('deterministic', True, 'deterministic mode')
  flags.DEFINE_string('problem', 'test', 'problem')

  flags.DEFINE_integer('seed', 42, 'random seed')
  flags.DEFINE_integer('list_size', 0, 'size of list used to increase batch size, 0 means no')
  flags.DEFINE_integer('plot_num', None, 'number of plot cases to tfboard')

  flags.DEFINE_list('train_data_dirs', [], 'directories of training data')
  flags.DEFINE_list('train_data_globs', ['train*'], 'filename glob patterns of training data')
  flags.DEFINE_string('train_config_filename', 'train_config.json', 'config file for training')
  flags.DEFINE_list('test_data_dirs', None, 'directories of testing data')
  flags.DEFINE_list('test_data_globs', ['test*'], 'filename glob patterns of testing data')
  flags.DEFINE_string('test_config_filename', None, 'config file for testing')

  flags.DEFINE_integer('demo_num', 5, 'the range of the number of demos')
  flags.DEFINE_integer('cond_len', 50, 'the range of the length of the condition')
  flags.DEFINE_integer('qoi_len', 50, 'the range of the length of the qoi')

  flags.DEFINE_string('preprocess', 'id', 'the method for preprocessing keys and values')
  flags.DEFINE_integer('k_dim', 2, 'the dim of the key')
  flags.DEFINE_integer('v_dim', 1, 'the dim of the value')
  flags.DEFINE_string('k_mode', 'naive', 'mode for keys') 
  flags.DEFINE_integer('query_len_max', 50, 'the max length of the query')
  flags.DEFINE_integer('qoi_v_dim', 1, 'the output dim of the model, also the dim of qoi_v ')

  flags.DEFINE_integer('hidden_dim', 128, 'the dim of the model')
  flags.DEFINE_integer('num_heads', 8, 'the number of heads in the model')
  flags.DEFINE_integer('num_layers', 4, 'the number of layers in the model')
  flags.DEFINE_string('initializer', 'glorot_uniform', 'the initializer for the transformers')

  flags.DEFINE_integer('train_batch_size', 32, 'batch size')
  flags.DEFINE_integer('train_shuffle_buffer_size', 1000, 'shuffle buffer size')
  flags.DEFINE_float('train_peak_lr', 0.0001, 'training peak learning rate')
  flags.DEFINE_float('train_end_lr', 0.0, 'training ending learning rate')
  flags.DEFINE_integer('train_warmup_percent', 10, 'training warmup percentage')
  flags.DEFINE_integer('train_decay_percent', 100, 'training decay percentage')
  flags.DEFINE_float('train_gnorm_clip', 1.0, 'training gradient global norm clip')
  flags.DEFINE_float('train_weight_decay', 0.0001, 'training weight decay')

  flags.DEFINE_integer('epochs', 20, 'total num of epochs')
  flags.DEFINE_integer('steps_per_epoch', 10000, 'steps per epoch')
  flags.DEFINE_integer('steps_per_loss', 100, 'steps per loss print')

  flags.DEFINE_enum('main', 'train', ['test','train'], 'train or test')

  app.run(main)

