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
    
    if model_name == 'deepo':
      import models_deepo
      self.model = models_deepo.DeepONet(model_config)
    elif model_name == 'fno':
      import models_fno
      self.model = models_fno.FNO(model_config)

    print(model_name, flush=True)  
    self.loss_mode = loss_mode
    print("loss_mode: {}".format(self.loss_mode), flush=True)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        self.model = torch.nn.DataParallel(self.model)
        print("model wrapped by DataParallel", flush=True)

    self.model.to(device)
    print('model moved to {}'.format(device), flush=True)

    self.opt_config = opt_config
    self.model_config = model_config

    model = self.model.module if hasattr(self.model, 'module') else self.model
    

    headers = ["Parameter Name", "Shape", "Requires Grad"]
    table_data = [(name, str(param.shape), param.requires_grad) for name, param in model.named_parameters()]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    headers = ["Trainable Parameters", "Shape"]
    table_data = [(name, str(param.shape)) for name, param in model.named_parameters() if param.requires_grad]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print(self.model, flush=True)
    print("# of trainable variables", sum(p.numel() for p in self.model.parameters() if p.requires_grad), flush=True)

    self.reset_optimizer()

  def reset_optimizer(self):

    model = self.model.module if hasattr(self.model, 'module') else self.model
    self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr = self.opt_config['peak_lr'], 
                                 weight_decay=self.opt_config['weight_decay'])
    self.lr_scheduler = utils.WarmupCosineDecayScheduler(optimizer=self.optimizer, 
                                                    warmup=self.opt_config['warmup_steps'],
                                                    max_iters=self.opt_config['decay_steps'],)
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
    loss = self.loss_fn(data, label)
    self.optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    model = self.model.module if hasattr(self.model, 'module') else self.model
    torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_config['gnorm_clip'])
    self.optimizer.step()
    self.lr_scheduler.step()
    self.train_step += 1

  
  def loss_fn(self, data, label):
    # assume data and label are already torch tensor on device, only handel one loss mode at a time
    if self.loss_mode == ['demo_quest']: # use both demo and quest
      cond_k = torch.concat([data.demo_cond_k, data.quest_cond_k], axis = 1) # [bs, num, len, dim]
      cond_v = torch.concat([data.demo_cond_v, data.quest_cond_v], axis = 1) # [bs, num, len, dim]
      qoi_k = torch.concat([data.demo_qoi_k, data.quest_qoi_k], axis = 1) # [bs, num, len, dim]
      qoi_v_gt = torch.concat([data.demo_qoi_v, label], axis = 1) # [bs, num, len, dim]
    elif self.loss_mode == ['demo']: # only use demo
      cond_k = data.demo_cond_k # [bs, num, len, dim]
      cond_v = data.demo_cond_v # [bs, num, len, dim]
      qoi_k = data.demo_qoi_k # [bs, num, len, dim]
      qoi_v_gt = data.demo_qoi_v # [bs, num, len, dim]
    else:
      raise ValueError('loss_mode {} not supported'.format(self.loss_mode))
    

    cond_k_flat = cond_k.reshape(-1, *cond_k.shape[2:]) # [bs*num, len, dim]
    cond_v_flat = cond_v.reshape(-1, *cond_v.shape[2:]) # [bs*num, len, dim]
    qoi_k_flat = qoi_k.reshape(-1, *qoi_k.shape[2:]) # [bs*num, len, dim]
    qoi_v_gt_flat = qoi_v_gt.reshape(-1, *qoi_v_gt.shape[2:]) # [bs*num, len, dim]

    output = self.model(cond_k_flat, cond_v_flat, qoi_k_flat) # (bs*num, query_len, dim)
    loss = torch.mean((output - qoi_v_gt_flat)**2)
    return loss
      
  def get_loss(self, data, label, loss_mode = None):
    # assume numpy data and label, return numpy loss
    data = inputdata_transform(lambda x: torch.tensor(x).to(device), data)
    label = torch.tensor(label).to(device)
    loss = self.loss_fn(data, label).detach().cpu().numpy()
    return loss
  
  def get_pred(self, data, with_caption=False):
    # assume numpy data, return numpy pred
    data = inputdata_transform(lambda x: torch.tensor(x).to(device), data)
    cond_k = data.quest_cond_k[:,0,:,:] # [bs, len, dim]
    cond_v = data.quest_cond_v[:,0,:,:] # [bs, len, dim]
    qoi_k = data.quest_qoi_k[:,0,:,:] # [bs, len, dim]
    out = self.model(cond_k, cond_v, qoi_k) # (bs, query_len, 1)
    return out.detach().cpu().numpy()
  
  def get_error(self, data, label, with_caption, return_pred = False, average_axis = (-1,)):
    # assume numpy data and label, return numpy error
    pred = self.get_pred(data, with_caption) # [bs, len, dim]
    qoi = label[:,0,:,:] # [bs, len, dim]
    error = np.linalg.norm(pred - qoi, axis = -1) # [batch_on_each_device, len]
    error = np.mean(error, axis = average_axis) # [batch_on_each_device] by default
    if return_pred:
      return error, pred
    else:
      return error


def test(loss_mode):
  import dataloader
  import numpy as np
  import utils
  from pprint import pprint

  data_config = utils.load_json('config_data/train_lm_pde_full_config.json')
  model_config = utils.load_json('config_model/model_deepo_pde_config.json')

  data_config['load_list'] = []

  data_provider = dataloader.DataProvider(seed = 1, config = data_config, 
                              file_names = '/home/shared/icon/data/data0910c/train_pde*',
                              batch_size = 2, shuffle_buffer_size = 100,
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
                  model_name = 'deepo', pretrained = True, 
                  trainable_mode = 'all', loss_mode = loss_mode)
  runner.iter(data, label)
  loss = runner.get_loss(data, label)
  pred = runner.get_pred(data, with_caption = False)
  error = runner.get_error(data, label, with_caption = False)
  print(loss, pred.shape, error.shape, flush=True)

if __name__ == "__main__":
  test(loss_mode = ['nocap', 'cap'])