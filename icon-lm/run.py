import torch
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')
from pprint import pprint
import jax.tree_util as tree
import numpy as np
import pytz
from datetime import datetime
import utils
from absl import app, flags, logging
import plot
from einshape import numpy_einshape as einshape

gpus = tf.config.list_physical_devices(device_type = 'GPU')
print(gpus, flush=True)

def run_train():
  utils.set_seed(FLAGS.seed)
  from dataloader import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader

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

  train_config = utils.load_json("config_data/" + FLAGS.train_config_filename)

  if FLAGS.test_config_filename is None:
    test_config = train_config.copy()
    test_config['select_caption'] = 'all' # test all the captions
    print('test_config = train_config, with select_caption = all', flush=True)
  else:
    test_config = utils.load_json("config_data/" + FLAGS.test_config_filename)

  if FLAGS.test_batch_size is None:
    test_batch_size = FLAGS.train_batch_size
  else:
    test_batch_size = FLAGS.test_batch_size

  if FLAGS.plot_num is None:
    plot_num = FLAGS.train_batch_size
  else:
    plot_num = FLAGS.plot_num
  
  test_demo_num_list = [int(i) for i in FLAGS.test_demo_num_list]
  if FLAGS.test_caption_id_list is None:
    test_caption_id_list = None
  else:
    test_caption_id_list = [int(i) for i in FLAGS.test_caption_id_list]


  model_config = utils.load_json("config_model/" + FLAGS.model_config_filename)

  real_time = FLAGS.real_time
  if 'cap' not in FLAGS.loss_mode:
    model_config['caption_len'] = 0
    train_config['load_list'] = []
    test_config['load_list'] = []
    real_time = True

  print("train_config: ", flush=True)
  pprint(train_config)
  print("test_config: ", flush=True)
  pprint(test_config)
  
  print('-----------------------model config-----------------------', flush=True)
  pprint(model_config)
  print('-----------------------model config end-----------------------', flush=True)

  if FLAGS.backend == 'jax':
    optimizer = utils.get_scheduled_adamw(peak_lr = FLAGS.train_peak_lr, 
                                          end_lr = FLAGS.train_end_lr,
                                          warmup_steps = train_warmup_steps,
                                          decay_steps = train_decay_steps,
                                          gnorm_clip = FLAGS.train_gnorm_clip,
                                          weight_decay = FLAGS.train_weight_decay)
    import jax
    data_num_devices = len(jax.devices())
  elif FLAGS.backend == 'torch':
    opt_config = {'peak_lr': FLAGS.train_peak_lr,
                  'end_lr': FLAGS.train_end_lr,
                  'warmup_steps': train_warmup_steps,
                  'decay_steps': train_decay_steps,
                  'gnorm_clip': FLAGS.train_gnorm_clip,
                  'weight_decay': FLAGS.train_weight_decay,
                  }
    data_num_devices = 0

  train_data = DataProvider(seed = FLAGS.seed + 1,
                            config = train_config,
                            file_names = train_file_names,
                            batch_size = FLAGS.train_batch_size,
                            shuffle_buffer_size = FLAGS.train_shuffle_buffer_size, 
                            deterministic = FLAGS.deterministic,
                            real_time = real_time,
                            name = 'train',
                            num_devices=data_num_devices,
                            )
  if FLAGS.vistest:
    test_data_with_train_config = DataProvider(seed = FLAGS.seed + 10,
                                              config = train_config,
                                              file_names = test_file_names,
                                              batch_size = FLAGS.train_batch_size,
                                              shuffle_buffer_size = FLAGS.train_shuffle_buffer_size, 
                                              deterministic = True,
                                              real_time = real_time,
                                              name = 'test',
                                              num_devices=data_num_devices,
                                              )
  test_data = DataProvider(seed = FLAGS.seed + 10,
                            config = test_config,
                            file_names = test_file_names,
                            batch_size = test_batch_size,
                            shuffle_buffer_size = FLAGS.train_shuffle_buffer_size, 
                            deterministic = True,
                            real_time = real_time,
                            name = 'test_full',
                            num_devices=data_num_devices,
                          )
  
  if FLAGS.vistest:
    datasets = [train_data, test_data_with_train_config]
  else:
    datasets = [train_data]
  
  equation, caption, data, label = train_data.get_next_data(caption_max_len = model_config['caption_len'])
  print_eqn_caption(equation, caption)
  print(tree.tree_map(lambda x: x.shape, data)) 

  if FLAGS.model in ['icon']:
    from runner_jax import Runner_vanilla
    runner = Runner_vanilla(seed = FLAGS.seed,
                    model = FLAGS.model,
                    data = data,
                    model_config = model_config,
                    optimizer = optimizer,
                    trainable_mode = FLAGS.trainable_mode,
                    )
  elif FLAGS.model in ['icon_lm']:
    from runner_jax import Runner_lm
    runner = Runner_lm(seed = FLAGS.seed,
                    model = FLAGS.model,
                    data = data,
                    model_config = model_config,
                    optimizer = optimizer,
                    trainable_mode = FLAGS.trainable_mode,
                    loss_mode = FLAGS.loss_mode,
                    )
  elif FLAGS.model in ['gpt2']:
    from runner_torch import Runner
    runner = Runner(data, model_config, opt_config = opt_config, 
                    model_name = FLAGS.model, pretrained = FLAGS.pretrained, 
                    trainable_mode = FLAGS.trainable_mode,
                    loss_mode = FLAGS.loss_mode,
                    )
  elif FLAGS.model in ['deepo','fno']:
    from runner_deepo_torch import Runner
    runner = Runner(data, model_config, opt_config = opt_config, 
                    model_name = FLAGS.model, pretrained = FLAGS.pretrained, 
                    trainable_mode = FLAGS.trainable_mode,
                    loss_mode = FLAGS.loss_mode,
                    )
  else:
    raise ValueError("model {} not supported".format(FLAGS.model))

  if FLAGS.restore_dir is not None:
    runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
    
  if FLAGS.tfboard:
    results_dir = f'/home/shared/icon/save/{FLAGS.user}/results/{FLAGS.problem}/'+ stamp
    file_writer = tf.summary.create_file_writer(results_dir)
    file_writer.set_as_default()
    ckpt_dir = f'/home/shared/icon/save/{FLAGS.user}/ckpts/{FLAGS.problem}/'+ stamp
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

  utils.timer.tic("since last print")
  for _ in range(FLAGS.epochs * FLAGS.steps_per_epoch + 1):
    if FLAGS.tfboard and runner.train_step % (FLAGS.save_freq) == 0:
      logging.info("current time: " + datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%Y%m%d-%H%M%S"))
      runner.save(ckpt_dir)
    
    # calculate, print and write loss to tfboard
    if FLAGS.profile_level == 0 and ((runner.train_step % FLAGS.loss_freq == 0)
        or (runner.train_step % (FLAGS.loss_freq//10) == 0 and runner.train_step <= FLAGS.loss_freq) 
        or (runner.train_step % (FLAGS.loss_freq//10) == 0 and runner.train_step >= FLAGS.epochs * FLAGS.steps_per_epoch - FLAGS.loss_freq)):
      print("==================== step: {}, loss start ====================".format(runner.train_step))
      utils.timer.toc("since last print")
      utils.timer.tic("since last print")
      utils.timer.tic("calculate, print and write loss to tfboard")

      for dataset in datasets:
        equation, caption, data, label = dataset.get_next_data(caption_max_len = model_config['caption_len'])
        this_loss = runner.get_loss(data, label)
        this_loss_mean, this_loss_std = np.mean(this_loss), np.std(this_loss)
        print("{} loss: {:.4f}+-{:.4f}".format(dataset.name, this_loss_mean, this_loss_std), flush=True)
        if FLAGS.tfboard:
          tf.summary.scalar(f'loss/{dataset.name}', this_loss_mean, step = runner.train_step)
          
      equation, caption, data, label = test_data.get_next_data(caption_max_len = model_config['caption_len'])
      for demo_num, caption_id, caption, data in split_data(caption, data, test_demo_num_list, test_caption_id_list):
        # without caption
        if caption_id == -1:
          this_error = runner.get_error(data, label, with_caption = False) # WARNING: it's actually the error for quest, not the loss
          this_error_mean, this_error_std = np.mean(this_error), np.std(this_error)
          print("test with demo num {}, no caption  , error: {:.4f}+-{:.4f}".format(
                  demo_num, this_error_mean, this_error_std), flush=True)
          if FLAGS.tfboard:
            tf.summary.scalar(f'test error, demo num: {demo_num}/no caption', this_error_mean, step = runner.train_step)
        # with caption
        else:
          this_error = runner.get_error(data, label, with_caption = True) # WARNING: it's actually the error for quest, not the loss
          this_error_mean, this_error_std = np.mean(this_error), np.std(this_error)
          print("test with demo num {}, caption id {}, error: {:.4f}+-{:.4f}".format(
                  demo_num, caption_id, this_error_mean, this_error_std), flush=True)
          if FLAGS.tfboard:
            tf.summary.scalar(f'test error, demo num: {demo_num}/caption id: {caption_id}', this_error_mean, step = runner.train_step)
      
      utils.timer.toc("calculate, print and write loss to tfboard")
      print("==================== step: {}, loss end ====================".format(runner.train_step))

     # make prediction and plot to tfboard
    if FLAGS.profile_level == 0 and plot_num > 0 and ((runner.train_step % FLAGS.plot_freq == 0)
        or (runner.train_step % (FLAGS.plot_freq//10) == 0 and runner.train_step <= FLAGS.plot_freq)
        or (runner.train_step % (FLAGS.plot_freq//10) == 0 and runner.train_step >= FLAGS.epochs * FLAGS.steps_per_epoch - FLAGS.plot_freq)):
      for dataset in [train_data, test_data]:
        print("==================== {} data: ==================== ".format(dataset.name), flush=True)
        for _ in range(FLAGS.print_eqn_caption_num):
          equation, caption, data, label = dataset.get_next_data(caption_max_len = model_config['caption_len'])
          print_eqn_caption(equation, caption, decode = False)
        print("==================== {} data: ==================== ".format(dataset.name), flush=True)

      utils.timer.tic("plot to tfboard")      
      for dataset in datasets:
        equation, caption, data, label = dataset.get_next_data(caption_max_len = model_config['caption_len'])
        pred = runner.get_pred(data, with_caption=False) # (num_devices, batch_on_each_device, ...)
        if FLAGS.tfboard:
          if dataset.num_devices > 0: # additional dimension for num_devices
            this_data = tree.tree_map(lambda x: einshape('ij...->(ij)...', np.array(x)), data)
            this_label = einshape('ij...->(ij)...', np.array(label))
            this_pred = einshape('ij...->(ij)...', np.array(pred))
          else: # no additional dimension for num_devices
            this_data = tree.tree_map(lambda x: np.array(x), data)
            this_label = np.array(label)
            this_pred = np.array(pred)
          for fij in range(plot_num):
            this_equation_ij = equation[fij] if type(equation[fij]) == str else equation[fij].numpy().decode('utf-8')
            this_caption_ij = caption[fij] if type(caption[fij]) == str else caption[fij].numpy().decode('utf-8')
            this_data_ij = tree.tree_map(lambda x: x[fij], this_data)
            figure = plot.plot_data(this_equation_ij, this_caption_ij,
                                    this_data_ij, this_label[fij], this_pred[fij], train_config)
            tf.summary.image("{} case {}".format(dataset.name, fij), figure, step = runner.train_step)
      utils.timer.toc("plot to tfboard")

    if FLAGS.profile_level <= 1:
      _, caption, data, label = train_data.get_next_data(caption_max_len = model_config['caption_len'])
      
    if FLAGS.train:
      runner.iter(data, label)
    else:
      runner.train_step += 1 # skip training

    if runner.train_step == 100: # exclude warming up steps
      utils.timer.tic("time estimate")
    if runner.train_step > 0 and (runner.train_step % FLAGS.time_freq == 0):
      ratio = (runner.train_step - 100)/(FLAGS.epochs * FLAGS.steps_per_epoch)
      samples_processed = (runner.train_step - 100) * FLAGS.train_batch_size
      utils.timer.estimate_time("time estimate", ratio, samples_processed)

    if FLAGS.profile_level >= 1 and runner.train_step == 3000:
      print("profile level {} reached, exiting...".format(FLAGS.profile_level))
      break
      
def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  tf.random.set_seed(FLAGS.seed + 123456) 
  if FLAGS.main == 'train':
    run_train()
  elif FLAGS.main == 'test':
    raise NotImplementedError


if __name__ == '__main__':

  FLAGS = flags.FLAGS
  flags.DEFINE_boolean('tfboard', False, 'dump into tfboard')

  flags.DEFINE_boolean('deterministic', True, 'deterministic mode')
  
  flags.DEFINE_string('user', 'user', 'user name, used for saving results and check points')
  flags.DEFINE_string('problem', 'test', 'folder for storing the model checkpoints and tensorboard logs')
  flags.DEFINE_enum('backend', 'jax', ['jax','torch'], 'backend of runner')

  flags.DEFINE_integer('seed', 42, 'random seed')

  flags.DEFINE_boolean('train', True, 'train the neural network')
  flags.DEFINE_integer('profile_level', 0, '0: usual training, 1: profile training, 2: profile without loading data')
  flags.DEFINE_boolean('vistest', False, 'visualize test dataset')
  flags.DEFINE_boolean('real_time', True, 'get caption in real time')
  flags.DEFINE_boolean('pretrained', True, 'use pretrained huggerface model')


  flags.DEFINE_list('train_data_dirs', [], 'directories of training data')
  flags.DEFINE_list('train_data_globs', ['train*'], 'filename glob patterns of training data')
  flags.DEFINE_string('train_config_filename', 'train_config.json', 'config file for training')
  flags.DEFINE_list('test_data_dirs', None, 'directories of testing data')
  flags.DEFINE_list('test_data_globs', ['test*'], 'filename glob patterns of testing data')
  flags.DEFINE_string('test_config_filename', 'test_config.json', 'config file for testing')
  flags.DEFINE_list('test_demo_num_list', [0,1,3,5], 'demo number list for testing')
  flags.DEFINE_list('test_caption_id_list', [-1,], 'caption id list for testing')

  flags.DEFINE_string('model', 'icon', 'model name')
  flags.DEFINE_string('model_config_filename', 'model_icon_cap_config.json', 'config file for model')

  flags.DEFINE_string('restore_dir', None, 'restore directory')
  flags.DEFINE_integer('restore_step', 1000000, 'restore step')
  flags.DEFINE_string('trainable_mode', 'all', 'trainable variables')
  flags.DEFINE_list('loss_mode', ['cap', 'nocap'], 'loss mode')

  flags.DEFINE_integer('train_batch_size', 32, 'batch size')
  flags.DEFINE_integer('test_batch_size', None, 'test batch size')
  flags.DEFINE_integer('train_shuffle_buffer_size', 1000, 'shuffle buffer size')
  flags.DEFINE_float('train_peak_lr', 0.0001, 'training peak learning rate')
  flags.DEFINE_float('train_end_lr', 0.0, 'training ending learning rate')
  flags.DEFINE_integer('train_warmup_percent', 10, 'training warmup percentage')
  flags.DEFINE_integer('train_decay_percent', 100, 'training decay percentage')
  flags.DEFINE_float('train_gnorm_clip', 1.0, 'training gradient global norm clip')
  flags.DEFINE_float('train_weight_decay', 0.0001, 'training weight decay')

  flags.DEFINE_integer('epochs', 100, 'total num of epochs')
  flags.DEFINE_integer('steps_per_epoch', 10000, 'steps per epoch')
  flags.DEFINE_integer('steps_per_loss', 100, 'steps per loss print')

  flags.DEFINE_integer('loss_freq', 1000, 'frequency of printing loss')
  flags.DEFINE_integer('save_freq', 100000, 'frequency of saving model')
  flags.DEFINE_integer('plot_freq', 10000, 'frequency of plotting to tfboard')
  flags.DEFINE_integer('time_freq', 1000, 'frequency of estimating time')
  flags.DEFINE_integer('plot_num', None, 'number of plot cases to tfboard')
  flags.DEFINE_integer('print_eqn_caption_num', 1, 'number of equations and captions to print')

  flags.DEFINE_enum('main', 'train', ['test','train'], 'train or test')

  app.run(main)

