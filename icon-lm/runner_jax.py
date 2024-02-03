import jax
import jax.numpy as jnp
import numpy as np
import pickle
import haiku as hk
import utils

from dataloader import Data

from einshape import jax_einshape as einshape
from absl import logging
import jax.tree_util as tree
from flax.core import FrozenDict


def update_dict(params1, params2):
  assert type(params1) == type(params2)
  if type(params1) == dict:
    params1.update(params2)
  elif type(params1) == FrozenDict:
    p = dict(params1['params'])
    p.update(params2['params'])
    params1 = {'params': FrozenDict(p)}
  else:
    raise ValueError('params1 type {} not implemented'.format(type(params1)))
  return params1

class Runner():
  def __init__(self):
    raise NotImplementedError
  
  def init_fn(self, optimizer, trainable_mode, devices):
    self.devices = devices
    self.num_devices = len(devices)
    if self.print_model:
      utils.print_pytree(self.params) # print the params

    if trainable_mode == 'all':
      print('+'*20, 'train all variables', '+'*20)
    elif trainable_mode == 'caption':
      print('+'*20, 'train caption-related variables only', '+'*20)
      optimizer = utils.get_partial_optimizer(params = self.params, trainable_key_list = ['caption'],
                                              untrainable_key_list = [], optimizer = optimizer)
      print('+'*20, 'train caption-related variables only', '+'*20)
    else:
      raise ValueError('trainable_mode {} not implemented'.format(trainable_mode))
    
    self.opt_state = optimizer.init(self.params) # intialize the optimizer

    # no multi-gpu yet
    self.predict_with_caption_batch_fn = jax.jit(jax.vmap(self.predict_with_caption_fn, in_axes = [None, 0, 0], out_axes = 0)) # predict in batch  
    self.predict_without_caption_batch_fn = jax.jit(jax.vmap(self.predict_without_caption_fn, in_axes = [None, 0, 0], out_axes = 0)) # predict in batch
    self.loss_batch_fn = jax.jit(jax.vmap(self.loss_fn, in_axes = [None, 0, 0, 0], out_axes = 0)) # no average over batch
    self.loss_batch_ave_fn = jax.jit(lambda *args, **kwargs: jnp.mean(self.loss_batch_fn(*args, **kwargs))) # average over batch
    
    # now work on the multi-gpu part
    self.params = jax.device_put_replicated(self.params, devices) # replicate the params to all devices
    self.opt_state = jax.device_put_replicated(self.opt_state, devices) # replicate the opt_state to all devices
    self.predict_with_caption_pmap_batch_fn = jax.pmap(self.predict_with_caption_batch_fn, axis_name='devices') # make predictions in batch and devices
    self.predict_without_caption_pmap_bath_fn = jax.pmap(self.predict_without_caption_batch_fn, axis_name='devices') # make predictions in batch and devices
    self.loss_pmap_batch_fn = jax.pmap(self.loss_batch_fn, axis_name='devices') # no average over batch and devices
    self.loss_pmap_batch_ave_fn = jax.pmap(self.loss_batch_ave_fn, axis_name='devices') # average over batch, no average over devices
    self.train_iter = utils.get_train_iter_pmap(self.loss_batch_ave_fn, optimizer)
    
    self.train_step = 0

  def _next_key(self, batch_on_each_device):
    '''split and duplicate the key for each device'''
    rng = jax.random.split(next(self.rng), self.num_devices * batch_on_each_device)
    return einshape('(ij)k->ijk', rng, i = self.num_devices, j = batch_on_each_device) # [num_devices, batch_on_each_device, 2]

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

    params_path = save_dir + '/{}_params.pickle'.format(step)
    opt_state_path = save_dir + '/{}_opt_state.pickle'.format(step)

    with open(params_path, 'rb') as file:
      params = pickle.load(file)
      replicate_params = jax.device_put_replicated(params, self.devices) # replicate the params to all devices
      self.params = update_dict(self.params, replicate_params)
    logging.info('restored params from {}, step {}'.format(save_dir, step))

    if restore_opt_state:
      with open(opt_state_path, 'rb') as file:
        opt_state = pickle.load(file)
        replicate_opt_state = jax.device_put_replicated(opt_state, self.devices) # replicate the opt_state to all devices
        self.opt_state = update_dict(self.opt_state, replicate_opt_state)
      logging.info('restored opt state from {}, step {}'.format(save_dir, step))
  
  def iter(self, data, label):
    '''data: (num_devices, batch_on_each_device, ...)'''
    batch_on_each_device = data.demo_cond_v.shape[1]
    self.params, self.opt_state = self.train_iter(self.params, self._next_key(batch_on_each_device), self.opt_state, data, label)
    self.train_step += 1

  def get_loss(self, data, label):
    '''data: (num_devices, batch_on_each_device, ...)
    return losses, no average over devices or batch'''
    batch_on_each_device = data.demo_cond_v.shape[1]
    losses = self.loss_pmap_batch_fn(self.params, self._next_key(batch_on_each_device), data, label) 
    return np.array(losses)

  def get_pred(self, data, with_caption):
    '''data: (num_devices, batch_on_each_device, ...)
    return prediction, no average over devices or batch'''
    batch_on_each_device = data.demo_cond_v.shape[1]
    if with_caption:
      pred = self.predict_with_caption_pmap_batch_fn(self.params, self._next_key(batch_on_each_device), data)
    else:
      pred = self.predict_without_caption_pmap_bath_fn(self.params, self._next_key(batch_on_each_device), data)
    return np.array(pred)
  
  def get_error(self, data, label, with_caption, return_pred = False, average_axis = (-1,)):
    '''data: (num_devices, batch_on_each_device, ...)
    by default average over keys, no average over devices or batch'''
    pred = self.get_pred(data, with_caption)
    error = np.linalg.norm(pred - label[:,:,0,:,:], axis = -1) # [num_devices, batch_on_each_device, len]
    error = np.mean(error, where = data.quest_qoi_mask[:,:,0,:], axis = average_axis) # [num_devices, batch_on_each_device] by default
    if return_pred:
      return error, pred
    else:
      return error


class Runner_vanilla(Runner):
  def __init__(self, seed, model, data, model_config, optimizer, trainable_mode, devices = jax.devices(), print_model = True):
    
    self.seed = seed
    self.print_model = print_model
    self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    if model == 'icon':
      import models_icon as models
      self.predict_fn, self.params = models.build_network_fn(data, next(self.rng), model_config)
    else:
      raise ValueError('model {} not implemented'.format(model))
    
    self.loss_fn = self._build_loss_fn(self.predict_fn) # no batch
    self.predict_with_caption_fn = self.predict_fn
    self.predict_without_caption_fn = self.predict_fn
    
    self.init_fn(optimizer, trainable_mode, devices)
  
  def _build_loss_fn(self, predict_fn):
    @jax.jit
    def loss_fn(params, rng_key, data, label):
      '''the loss function without batch'''
      out = predict_fn(params, rng_key, data)
      loss = jnp.mean((out - label[0,:,:])**2, where = data.quest_qoi_mask[0,:,None])
      return loss
    return loss_fn


class Runner_lm(Runner):
  def __init__(self, seed, model, data, model_config, optimizer, trainable_mode, loss_mode, devices = jax.devices(), print_model = True):
    
    self.seed = seed
    self.print_model = print_model
    self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

    if model == 'icon_lm':
      import models_lm as models
      forward_with_caption_fn, forward_without_caption_fn, self.predict_with_caption_fn, self.predict_without_caption_fn, self.params = \
              models.build_network_fn(data, next(self.rng), model_config, print_model = print_model)
    else:
      raise ValueError('model {} not implemented'.format(model))
    
    self.loss_fn = self._build_loss_fn(forward_with_caption_fn, forward_without_caption_fn,
                                       self.predict_with_caption_fn, self.predict_without_caption_fn,
                                       loss_mode) # no batch
    self.init_fn(optimizer, trainable_mode, devices)
  
  def _build_loss_fn(self, forward_with_caption_fn, forward_without_caption_fn, predict_with_caption_fn, predict_without_caption_fn, loss_mode):
    def _build_gt_and_mask(data, label, shot_num_min):
      '''build the ground truth and mask for the loss function'''
      ground_truth = einshape('nld->(nl)d', data.demo_qoi_v[shot_num_min:,:,:] ) # [num, len, dim] -> [num*len, dim], remove the first one
      ground_truth = jnp.concatenate([ground_truth, label[0,:,:]], axis = 0) # [num*len+quest_qoi_len, dim]
      mask = einshape('nl->(nl)', data.demo_qoi_mask[shot_num_min:,:] ) # [num, len] -> [num*len], remove the first one
      mask = jnp.concatenate([mask, data.quest_qoi_mask[0,:]], axis = 0) # [num*len+quest_qoi_len]
      return ground_truth, mask
    
    def _build_concate(data, label):
      cond_k = jnp.concatenate([data.demo_cond_k, data.quest_cond_k], axis = 0) # [demo_num+1, demo_cond_len, k_dim]
      cond_v = jnp.concatenate([data.demo_cond_v, data.quest_cond_v], axis = 0) # [demo_num+1, demo_cond_len, v_dim]
      cond_mask = jnp.concatenate([data.demo_cond_mask, data.quest_cond_mask], axis = 0) # [demo_num+1, demo_cond_len]
      qoi_k = jnp.concatenate([data.demo_qoi_k, data.quest_qoi_k], axis = 0) # [demo_num+1, demo_qoi_len, k_dim]
      qoi_v = jnp.concatenate([data.demo_qoi_v, label], axis = 0) # [demo_num+1, demo_qoi_len, v_dim]
      qoi_mask = jnp.concatenate([data.demo_qoi_mask, data.quest_qoi_mask], axis = 0) # [demo_num+1, demo_qoi_len]
      return cond_k, cond_v, cond_mask, qoi_k, qoi_v, qoi_mask
    
    def _build_except(cond_k, cond_v, cond_mask, qoi_k, qoi_v, qoi_mask, demo_num):
      # suppose len(cond_k) = 6, exclude the demo_num-th one
      if demo_num < len(cond_k) - 1: # 0,1,2,3,4
        cond_k_except = jnp.concatenate([cond_k[:demo_num], cond_k[demo_num+1:]], axis = 0) # [demo_num-1, demo_qoi_len, k_dim]
        cond_v_except = jnp.concatenate([cond_v[:demo_num], cond_v[demo_num+1:]], axis = 0) # [demo_num-1, demo_qoi_len, v_dim]
        cond_mask_except = jnp.concatenate([cond_mask[:demo_num], cond_mask[demo_num+1:]], axis = 0) # [demo_num-1, demo_qoi_len]
        qoi_k_except = jnp.concatenate([qoi_k[:demo_num], qoi_k[demo_num+1:]], axis = 0) # [demo_num-1, demo_qoi_len, k_dim]
        qoi_v_except = jnp.concatenate([qoi_v[:demo_num], qoi_v[demo_num+1:]], axis = 0) # [demo_num-1, demo_qoi_len, v_dim]
        qoi_mask_except = jnp.concatenate([qoi_mask[:demo_num], qoi_mask[demo_num+1:]], axis = 0) # [demo_num-1, demo_qoi_len]
      else: # 5
        cond_k_except = cond_k[:demo_num]
        cond_v_except = cond_v[:demo_num]
        cond_mask_except = cond_mask[:demo_num]
        qoi_k_except = qoi_k[:demo_num]
        qoi_v_except = qoi_v[:demo_num]
        qoi_mask_except = qoi_mask[:demo_num]
      return cond_k_except, cond_v_except, cond_mask_except, qoi_k_except, qoi_v_except, qoi_mask_except
    
    def _loss_fn_nocap(params, rng_key, data, label):
      out = forward_without_caption_fn(params, rng_key, data) 
      ground_truth, mask = _build_gt_and_mask(data, label, shot_num_min = 1)
      loss_nocap = jnp.mean((out - ground_truth)**2, where = mask[:,None])
      return loss_nocap

    def _loss_fn_cap(params, rng_key, data, label):
      out = forward_with_caption_fn(params, rng_key, data) 
      ground_truth, mask = _build_gt_and_mask(data, label, shot_num_min = 0)
      loss_cap = jnp.mean((out - ground_truth)**2, where = mask[:,None])
      return loss_cap
    
    def _inverse_forward(params_stop_gradient, rng_key, inverse_out, cond_k, cond_v, cond_mask, qoi_k, qoi_v, qoi_mask, demo_num):
      # inverse_out: [6, len, dim], prediction of the original condition

      cond_k_except, cond_v_except, cond_mask_except, qoi_k_except, qoi_v_except, qoi_mask_except = \
              _build_except(cond_k, cond_v, cond_mask, qoi_k, qoi_v, qoi_mask, demo_num)
      
      forward_data = Data(input_id = None, embedding_raw = None, embedding_pool = None, embedding_mask = None,
                          demo_cond_k = cond_k_except, demo_cond_v = cond_v_except, demo_cond_mask = cond_mask_except,
                          demo_qoi_k = qoi_k_except, demo_qoi_v = qoi_v_except, demo_qoi_mask = qoi_mask_except,
                          quest_cond_k = cond_k[demo_num:demo_num+1],
                          quest_cond_v = inverse_out[demo_num:demo_num+1],
                          quest_cond_mask = cond_mask[demo_num:demo_num+1],
                          quest_qoi_k = qoi_k[demo_num:demo_num+1],
                          quest_qoi_mask = qoi_mask[demo_num:demo_num+1])
      # this is prediction of the original qoi
      forward_out = predict_without_caption_fn(params_stop_gradient, rng_key, forward_data)
      return forward_out # [len, dim]
    
    def _loss_fn_consist(params, rng_key, data, label):
      cond_k, cond_v, cond_mask, qoi_k, qoi_v, qoi_mask = _build_concate(data, label)
      # exchange cond and qoi
      inverse_data = Data(input_id = None, embedding_raw = None, embedding_pool = None, embedding_mask = None,
                          demo_cond_k = data.demo_qoi_k, demo_cond_v = data.demo_qoi_v, demo_cond_mask = data.demo_qoi_mask,
                          demo_qoi_k = data.demo_cond_k, demo_qoi_v = data.demo_cond_v, demo_qoi_mask = data.demo_cond_mask,
                          quest_cond_k = data.quest_qoi_k,
                          quest_cond_v = label,
                          quest_cond_mask = data.quest_qoi_mask,
                          quest_qoi_k = data.quest_cond_k,
                          quest_qoi_mask = data.quest_cond_mask
                          )
      # this is prediction of the original condition, need forward_with_caption_fn since the first one should be kept
      inverse_out = forward_with_caption_fn(params, rng_key, inverse_data) # [num * len, dim]
      inverse_out = einshape('(nl)d->nld', inverse_out, n = len(cond_k)) # [num, len, dim]

      inverse_forward_outs = []
      params_stop_gradient = jax.tree_map(lambda x: jax.lax.stop_gradient(x), params)

      for demo_num in range(1,len(cond_k)):
        inverse_forward_out_i = _inverse_forward(params_stop_gradient, rng_key, inverse_out, 
                                                 cond_k, cond_v, cond_mask, qoi_k, qoi_v, qoi_mask, demo_num)
        inverse_forward_outs.append(inverse_forward_out_i)
      inverse_forward_outs = jnp.stack(inverse_forward_outs, axis = 0) # [num-1, len, dim]
      loss_consist = jnp.mean((inverse_forward_outs - qoi_v[1:])**2, where = qoi_mask[1:,:,None])
      return loss_consist
    
    if "consist" in loss_mode:
      @jax.jit
      def loss_fn(params, rng_key, data, label):
        loss_nocap = _loss_fn_nocap(params, rng_key, data, label)
        loss_consist = _loss_fn_consist(params, rng_key, data, label)
        return loss_nocap + loss_consist
      print("+"*20, "train with consistency loss", "+"*20)
      return loss_fn
    
    if ('cap' in loss_mode) and ('nocap' in loss_mode):
      @jax.jit
      def loss_fn(params, rng_key, data, label):
        loss_nocap = _loss_fn_nocap(params, rng_key, data, label)
        loss_cap = _loss_fn_cap(params, rng_key, data, label)
        return loss_nocap + loss_cap
      print('+'*20, 'train with caption and without caption', '+'*20)
    elif ('nocap' in loss_mode) and ('cap' not in loss_mode):
      loss_fn = jax.jit(_loss_fn_nocap)
      print('+'*20, 'train without caption', '+'*20)
    else:
      raise ValueError('loss_mode {} not implemented'.format(loss_mode))
    return loss_fn
