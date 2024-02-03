from functools import wraps, partial
import time
import haiku as hk
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import jax.tree_util as tree
import os
import json
import pytz
from datetime import datetime, timedelta
import re
import numpy as np
import torch.optim as optim
import random
import torch

# see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyles = {
     'solid': 'solid',
     'dotted': 'dotted',
     'dashed': 'dashed', 
     'dashdot': 'dashdot', 
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# concatenate a list of pytree along axis
concate_pytree_fn = lambda list_of_trees, axis = 0: tree.tree_map(lambda *xs: jnp.concatenate(list(xs), axis), *list_of_trees)
# transpose a list of pytree and convert to array
transpose_fn = lambda list_of_trees: tree.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)

def strip_ansi_codes(s):
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', s)

def get_git_hash():
  import subprocess
  return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_sentence_from_ids(ids, tokenizer, clean = True):
  tokens = tokenizer.convert_ids_to_tokens(ids)
  if clean:
    tokens_clean = [token for token in tokens if token not in tokenizer.all_special_tokens]
    sentence = tokenizer.convert_tokens_to_string(tokens_clean).replace(' ##', '')
  else:
    sentence = tokenizer.convert_tokens_to_string(tokens)
  return tokens, sentence

def find_sublist(a, b):
    a = list(a)
    b = list(b)
    len_a, len_b = len(a), len(b)
    for i in range(len_b - len_a + 1):
        if b[i:i + len_a] == a:
            return i + len_a  # add the length of a to the index
    return -1


def load_json(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def print_dot(i, freq = 100, marker = "."):
  if i % freq == 0:
    print(i, end = "", flush = True)
  print(marker, end = "", flush=True)
  if (i+1) % freq == 0:
    print("", flush=True)
    
def print_pytree(v, name = "trainable variables"):
    print('==================================================================')
    print('# {}:'.format(name), sum(x.size for x in tree.tree_leaves(v)), flush=True)
    shape = tree.tree_map(lambda x: x.shape, v)
    for key in shape:
      print(key, shape[key])
    print('# {}:'.format(name), sum(x.size for x in tree.tree_leaves(v)), flush=True)
    print('==================================================================')


class MLPLN(hk.Module):
  def __init__(self, widths, name = None, layer_norm=False):
    '''MLP with layer normalization'''
    super().__init__(name)
    self.network = hk.nets.MLP(widths, activate_final=False)
    if layer_norm:
      self.network = hk.Sequential([self.network, 
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
  def __call__(self,x):
    return self.network(x)
    

def get_positional_encoding(max_seq_length, d_model):
    """
    Compute the sinusoidal positional encoding for a given sequence length and model size using JAX.
    
    Parameters:
    - max_seq_length: The maximum length of the sequence
    - d_model: The depth of the model or embedding size

    Returns:
    - pos_encoding: (max_seq_length, d_model) JAX array with positional encodings
    """
    
    # Ensure the model size is even
    assert d_model % 2 == 0

    # Compute the position encodings
    pos = jnp.arange(max_seq_length)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pos_encoding = jnp.zeros((max_seq_length, d_model))
    
    pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(pos * div_term))
    pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(pos * div_term))
    
    return pos_encoding

def get_causal_mask(seq_len):
    """
    Create a causal mask for a sequence of a given length.

    Args:
    - seq_len (int): Length of the sequence.

    Returns:
    - jnp.ndarray: A causal mask of shape (seq_len, seq_len).
    """
    # Create an lower triangular matrix
    mask = jnp.tril(jnp.ones((seq_len, seq_len)), k=0)
    return mask

def timeit_full(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper

def get_days_hours_mins_seconds(time_consumed_in_seconds):
    time_consumed = time_consumed_in_seconds
    days_consumed = int(time_consumed // (24 * 3600))
    time_consumed %= (24 * 3600)
    hours_consumed = int(time_consumed // 3600)
    time_consumed %= 3600
    minutes_consumed = int(time_consumed // 60)
    seconds_consumed = int(time_consumed % 60)
    return days_consumed, hours_consumed, minutes_consumed, seconds_consumed

class TicToc:
  def __init__(self):
    self.start_time = {}
    self.end_time = {}
  def tic(self, name):
    self.start_time[name] = time.perf_counter()
  def toc(self, name):
    self.end_time[name] = time.perf_counter()
    total_time = self.end_time[name] - self.start_time[name]
    print(f'{name} Took {total_time:.4f} seconds', flush = True)
  def estimate_time(self, name, ratio, samples_processed = None, timezone_str='America/Los_Angeles'):
    print('==========================Time Estimation Starts==========================')
    timezone = pytz.timezone(timezone_str)
    current_time = datetime.now(timezone)
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current time in {timezone_str}:", current_time_str)
    self.end_time[name] = time.perf_counter()
    time_consumed = self.end_time[name] - self.start_time[name]
    days_consumed, hours_consumed, minutes_consumed, seconds_consumed = get_days_hours_mins_seconds(time_consumed)
    print(f"Time consumed: {days_consumed}-{hours_consumed:02d}:{minutes_consumed:02d}:{seconds_consumed:02d}")
    if samples_processed is not None:
      samples_processed_per_second = samples_processed / time_consumed
      print(f"Samples processed per second: {samples_processed_per_second:.2f}")
    time_remaining = time_consumed * (1 - ratio) / ratio
    days_remaining, hours_remaining, minutes_remaining, seconds_remaining = get_days_hours_mins_seconds(time_remaining)
    print(f"Estimated remaining time: {days_remaining}-{hours_remaining:02d}:{minutes_remaining:02d}:{seconds_remaining:02d}")
    time_total = time_consumed / ratio
    days_total, hours_total, minutes_total, seconds_total = get_days_hours_mins_seconds(time_total)
    print(f"Estimated total time: {days_total}-{hours_total:02d}:{minutes_total:02d}:{seconds_total:02d}")
    finish_time = current_time + timedelta(seconds=time_remaining)
    finish_time_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Estimated finishing time in {timezone_str}:", finish_time_str)
    print('==========================Time Estimation Ends==========================', flush=True)


timer = TicToc()


class WarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
          lr_factor = epoch * 1.0 / self.warmup
        else:
          progress = (epoch - self.warmup) / (self.max_num_iters - self.warmup)
          lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return lr_factor
    

def get_scheduled_adamw(peak_lr, end_lr, warmup_steps, decay_steps, gnorm_clip, weight_decay):
  lr_schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=peak_lr,
      warmup_steps=warmup_steps,
      decay_steps=decay_steps,
      end_value=end_lr
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(gnorm_clip),  # Clip gradients at norm 1
      optax.adamw(lr_schedule, weight_decay=weight_decay)
  )
  return optimizer

def get_train_iter(params, loss_fun, optimizer = None):
  if optimizer is None:
    optimizer = optax.adam(learning_rate=1e-4)
  opt_state = optimizer.init(params)
  d_loss_d_theta = jax.jit(jax.grad(loss_fun))

  # @jax.jit
  def train_iter(params, rng_key, opt_state, *args):
    grads = d_loss_d_theta(params, rng_key, *args)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
  
  return opt_state, train_iter


def get_partial_optimizer(params, trainable_key_list, untrainable_key_list, optimizer = None):
  '''
  trainable: in trainable_key_list, but not in untrainable_key_list
  '''
  def is_in_list(key):
    flag = "zero" # default to be untrainable
    for substr in trainable_key_list:
      if substr in key:
        flag="optimizer"
    for substr in untrainable_key_list:
      if substr in key:
        flag="zero"
    return flag
  
  param_labels = {}
  for key, value_dict in params.items():
    param_labels[key] = {k: is_in_list(key) for k in value_dict}

  for key in param_labels:
    print(key, param_labels[key])

  if optimizer is None:
    optimizer = optax.adam(learning_rate=1e-4)
  
  partial_optimizer = optax.multi_transform(
        {'optimizer': optimizer, 'zero': optax.set_to_zero()}, param_labels)
  return partial_optimizer

def get_train_iter_batch(params, loss_fn_batch, optimizer = None, averaged = True):
  '''
  get the train_iter function based on the loss function in batch and optimizer
  @params:
    params: neural network parameters
    loss_fn_batch: loss function in batch, return a scalar or a vector
    optimizer: optimizer for training
    averaged: whether loss_fn_batch returns the averaged loss (scalar) or not
  @return:
    opt_state: the initial state of the optimizer
    train_iter: training iteration function, need to jit outside
  '''
  if optimizer is None:
    optimizer = optax.adamw(learning_rate=1e-4)
  opt_state = optimizer.init(params)
  if not averaged:
    loss_fn_mean = lambda *args, **kwargs: jnp.mean(loss_fn_batch(*args, **kwargs))
  else:
    loss_fn_mean = loss_fn_batch
  # @jax.jit
  def train_iter(params, rng_key, opt_state, *args, **kwargs):
    grads = jax.grad(loss_fn_mean)(params, rng_key, *args, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
  
  return opt_state, train_iter


def get_train_iter_pmap(loss_batch_ave_fn, optimizer):
  '''
  get the train_iter function based on the loss function in batch and optimizer
  @params:
    loss_batch_ave_fn: average loss function in batch, return a scalar
    optimizer: optimizer for training
  @return:
    train_iter: training iteration function
  '''
  @partial(jax.pmap, axis_name='devices')
  def train_iter(params, rng_key, opt_state, *args, **kwargs):
    grads = jax.grad(loss_batch_ave_fn)(params, rng_key, *args, **kwargs) # gradient on each device
    grads = jax.lax.pmean(grads, axis_name='devices') # average the gradients over devices
    updates, opt_state = optimizer.update(grads, opt_state, params) # update on each device, with synchronized gradients
    params = optax.apply_updates(params, updates) # update parameters on each device
    return params, opt_state
  return train_iter


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

class DataIndex():
  '''
  dataset for index
  '''
  def __init__(self, index_list, seed):
    self.init_index_list = index_list
    self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
    self.index_list = self.init_index_list.copy()

  def next(self):
    if len(self.index_list) == 0:
      self.index_list = self.init_index_list.copy()
    i = jax.random.choice(next(self.rng), len(self.index_list)) # get random index
    self.index_list[i], self.index_list[-1] = self.index_list[-1], self.index_list[i]  # swap with the last element
    x = self.index_list.pop()                  # pop last element O(1)
    return x

class DataSet():
  '''
  DataSet for training, using tf dataset
  '''
  def __init__(self, data_list, buffer_cap, key, batch_size):
    '''
    data_list: a list of arrays, with the same size in axis = 0
              elements are different components of data, e.g. u, left bound, right bound 
              should be aligned
    buffer_cap: the capacity of buffer
    '''
    self.buffer_cap = buffer_cap
    self.key = key
    self.batch_size = batch_size
    self.buffer = [jax.device_put(d, jax.devices("cpu")[0]) for d in data_list]
    self.build_data_from_buffer()

  def add_to_buffer(self, new_data_list):

    buffer = [jnp.concatenate([jax.device_put(d, jax.devices("cpu")[0]),
                              jax.device_put(nd, jax.devices("cpu")[0])], axis = 0)[-self.buffer_cap:,...] 
                              for d, nd in zip(self.buffer, new_data_list)]
    self.buffer = buffer
    self.build_data_from_buffer()

  @timeit
  def build_data_from_buffer(self):
    self.key, subkey = jax.random.split(self.key)
    buffer = tuple(self.buffer)
    self.dataset = tf.data.Dataset.from_tensor_slices(buffer)
    self.dataset = self.dataset.shuffle(self.buffer_cap, seed = subkey[0])
    self.dataset = self.dataset.batch(self.batch_size, drop_remainder = True)
    self.dataset = self.dataset.repeat(None)
    self.iterator = iter(self.dataset)
  
  def next(self, gpu = True):
    out = next(self.iterator)
    if gpu:
      return [jax.device_put(o.numpy(), jax.devices("gpu")[0]) for o in out]
    else:
      return [o.numpy() for o in out]


if __name__ == "__main__":
  key = jax.random.PRNGKey(42)
  x = jax.random.uniform(key, (100,40))
  y = jnp.sum(x, axis = 1)
  
  dataset = DataSet([x,y], 150, jax.random.PRNGKey(0), batch_size = 20)
  dataset.add_to_buffer([x*1.1,y*1.1])
  xp, yp = dataset.next(gpu=False)
  assert jnp.sum((jnp.sum(xp, axis=1) - yp)**2) < 1e-6
  dataset.add_to_buffer([x*2,y*2])
  xp, yp = dataset.next(gpu=False)
  assert jnp.sum((jnp.sum(xp, axis=1) - yp)**2) < 1e-6
  print_pytree({"a":{"w":jnp.ones((1,2)), "b":jnp.ones((3,4))}})