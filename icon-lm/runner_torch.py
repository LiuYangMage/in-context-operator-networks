import torch
from absl import logging
import utils
import torch.optim as optim
import numpy as np
from tabulate import tabulate
from models_utils_pytorch import InputData, inputdata_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Runner():
  def __init__(self, data, model_config, opt_config, model_name, pretrained, trainable_mode, loss_mode):
    
    import models_gpt2_icon
    data_shape = inputdata_transform(lambda x: x[0,...].shape, data)
    self.model = models_gpt2_icon.ICONGPT2(model_config, data_shape, model_name, pretrained)
  
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        self.model = torch.nn.DataParallel(self.model)
        print("model wrapped by DataParallel", flush=True)

    self.model.to(device)
    print('model moved to {}'.format(device), flush=True)

    self.opt_config = opt_config
    self.model_config = model_config
    self.loss_mode = loss_mode  
    print("loss_mode: {}".format(self.loss_mode), flush=True)

    model = self.model.module if hasattr(self.model, 'module') else self.model
    if not (trainable_mode == 'all'): # freeze the model first
      for param in model.parameters():
        param.requires_grad = False
    
    # Dictionary mapping the component to its parameter name pattern
    patterns = {
        'data': ["input_layer", "output_layer", "func_pos_embedding"],
        'transformer': [".h.", ".ln_f."],
        'wte': [".wte."],
        'wpe': [".wpe."]
    }

    for name, params in model.named_parameters():
      for mode, pattern_list in patterns.items():
        if any(pattern in name for pattern in pattern_list) and mode in trainable_mode:
          params.requires_grad = True

    headers = ["Parameter Name", "Shape", "Requires Grad"]
    table_data = [(name, str(param.shape), param.requires_grad) for name, param in model.named_parameters()]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    headers = ["Trainable Parameters", "Shape"]
    table_data = [(name, str(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = opt_config['peak_lr'], weight_decay=opt_config['weight_decay'])
    self.lr_scheduler = utils.WarmupCosineDecayScheduler(optimizer=self.optimizer, 
                                                    warmup=opt_config['warmup_steps'],
                                                    max_iters=opt_config['decay_steps'],)
    print(self.model, flush=True)
    self.train_step = 0

  def save(self, save_dir):
    model = self.model.module if hasattr(self.model, 'module') else self.model
    torch.save(model.state_dict(), '{}/{}_params.pth'.format(save_dir, self.train_step))
    logging.info('saved to {}, step {}'.format(save_dir, self.train_step))

  def restore(self, save_dir, step, restore_opt_state = True):
    params_path = '{}/{}_params.pth'.format(save_dir, step)
    model = self.model.module if hasattr(self.model, 'module') else self.model
    model.load_state_dict(torch.load(params_path, map_location=device))
    logging.info('restored params from {}, step {}'.format(save_dir, step))

  
  def iter(self, data, label):
    '''data: (num_devices, batch_on_each_device, ...)'''
    # Compute loss
    data = inputdata_transform(lambda x: torch.tensor(x).to(device), data)
    label = torch.tensor(label).to(device)
    loss = 0
    for mode in self.loss_mode:
      loss += self.loss_fn(data, label, mode)
    self.optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    model = self.model.module if hasattr(self.model, 'module') else self.model
    torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_config['gnorm_clip'])
    self.optimizer.step()
    self.lr_scheduler.step()
    self.train_step += 1

  def _build_gt_and_mask(self, data, label, shot_num_min):
    '''build the ground truth and mask for the loss function'''
    # sequence the first two dimensions or torch tensor
    ground_truth = data.demo_qoi_v[:,shot_num_min:,:,:].reshape(data.demo_qoi_v.shape[0], -1, data.demo_qoi_v.shape[-1]) # [bs, num, len, dim] -> [bs, num*len, dim]
    ground_truth = torch.concatenate([ground_truth, label[:,0,:,:]], axis = 1) # [bs, num*len+quest_qoi_len, dim]
    
    mask = data.demo_qoi_mask[:,shot_num_min:,:].reshape(data.demo_qoi_mask.shape[0], -1) # [bs, num, len] -> [bs, num*len]
    mask = torch.concatenate([mask, data.quest_qoi_mask[:,0,:]], axis = 1) # [bs, num*len+quest_qoi_len]
    return ground_truth, mask

  def model_forward(self, data, mode):
    '''
    a wrapper to call model.forward
    '''
    input_id = data.input_id
    embedding_mask = data.embedding_mask
    demo_cond_k = data.demo_cond_k
    demo_cond_v = data.demo_cond_v
    demo_cond_mask = data.demo_cond_mask
    demo_qoi_k = data.demo_qoi_k
    demo_qoi_v = data.demo_qoi_v
    demo_qoi_mask = data.demo_qoi_mask
    quest_cond_k = data.quest_cond_k
    quest_cond_v = data.quest_cond_v
    quest_cond_mask = data.quest_cond_mask
    quest_qoi_k = data.quest_qoi_k
    quest_qoi_mask = data.quest_qoi_mask

    output = self.model.forward(input_id, embedding_mask,
                                demo_cond_k, demo_cond_v, demo_cond_mask,
                                demo_qoi_k, demo_qoi_v, demo_qoi_mask,
                                quest_cond_k, quest_cond_v, quest_cond_mask,
                                quest_qoi_k, quest_qoi_mask,
                                mode = mode)
    return output
  
  
  def loss_fn(self, data, label, loss_mode):
    # assume data and label are already torch tensor on device, only handel one loss mode at a time
    if loss_mode == "nocap":
      ground_truth, mask = self._build_gt_and_mask(data, label, 1)
      output = self.model_forward(data, mode = "forward_without_caption")
    elif loss_mode == "cap":
      ground_truth, mask = self._build_gt_and_mask(data, label, 0)
      output = self.model_forward(data, mode = "forward_with_caption")
    else:
      raise ValueError("loss_mode {} not supported".format(loss_mode))
    loss = torch.sum((output - ground_truth)**2 * mask[:,:,None])/torch.sum(mask)
    return loss
      
  def get_loss(self, data, label, loss_mode = None):
    # assume numpy data and label, return numpy loss
    data = inputdata_transform(lambda x: torch.tensor(x).to(device), data)
    label = torch.tensor(label).to(device)
    if loss_mode is None:
      loss_mode = self.loss_mode
    loss = [self.loss_fn(data, label, mode).detach().cpu().numpy() for mode in loss_mode]
    loss = np.sum(loss)
    return loss
  
  def get_pred(self, data, with_caption):
    # assume numpy data, return numpy pred
    data = inputdata_transform(lambda x: torch.tensor(x).to(device), data)
    if with_caption:
      out = self.model_forward(data, mode = self.model_config["caption_len"])
    else:
      out = self.model_forward(data, mode = 0)
    return out.detach().cpu().numpy()
  
  def get_error(self, data, label, with_caption, return_pred = False, average_axis = (-1,)):
    # assume numpy data and label, return numpy error
    pred = self.get_pred(data, with_caption)
    error = np.linalg.norm(pred - label[:,0,:,:], axis = -1) # [batch_on_each_device, len]
    error = np.mean(error, where = data.quest_qoi_mask[:,0,:], axis = average_axis) # [batch_on_each_device] by default
    if return_pred:
      return error, pred
    else:
      return error


def test(loss_mode):
  import dataloader
  import numpy as np
  import utils
  from pprint import pprint

  data_config = utils.load_json('config_data/train_lm_config.json')
  model_config = utils.load_json('config_model/model_gpt2_config.json')
  model_config['index_mode'] = 'learn'

  if "cap" in loss_mode:
    model_config['caption_len'] = 300
    data_config['load_list'] = ['input_id']
  else:
    model_config['caption_len'] = 0
    data_config['load_list'] = []

  data_provider = dataloader.DataProvider(seed = 1, config = data_config, 
                              file_names = '/home/shared/icon/data/data0910c/train*',
                              batch_size = 2, shuffle_buffer_size = 10,
                              num_devices = 0, real_time = True)
  
  equation, caption, data, label = data_provider.get_next_data()
  dataloader.print_eqn_caption(equation, caption, decode = False)
  pprint(inputdata_transform(lambda x: (type(x), x.shape), data)) 

  opt_config = {'peak_lr': 0.001,
                'end_lr': 0,
                'warmup_steps': 100,
                'decay_steps': 10000,
                'gnorm_clip': 1,
                'weight_decay': 0.1,
                }
        
  runner = Runner(data, model_config, opt_config = opt_config, 
                  model_name = 'gpt2', pretrained = True, 
                  trainable_mode = 'data_wpe', loss_mode = loss_mode)
  runner.iter(data, label)
  loss = runner.get_loss(data, label)
  pred = runner.get_pred(data, with_caption = False)
  error = runner.get_error(data, label, with_caption = False)
  print(loss, pred.shape, error.shape, flush=True)

if __name__ == "__main__":
  test(loss_mode = ['nocap', 'cap'])