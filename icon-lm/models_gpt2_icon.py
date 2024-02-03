import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Config
from models_gpt2_source import GPT2Model
from torch.optim import Adam
from models_utils_pytorch import build_bool_sequence, build_basic_mask, build_index_integer, build_out_mask
from models_utils_pytorch import build_data_sequence_batch, build_data_mask_batch
from models_utils_pytorch import InputData, inputdata_transform
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_matrices_from_data_shape(data_shape, config, compact, mode, caption_len, shot_num_min, return_shape_list=False):
    '''
    data_shape is the shape of data, usually obtained by torch.Tensor.size() or some dictionary-like structure.
    '''
    demo_num = data_shape.demo_cond_k[0]
    demo_cond_len = data_shape.demo_cond_k[1]
    demo_qoi_len = data_shape.demo_qoi_k[1]
    quest_cond_len = data_shape.quest_cond_k[1]
    quest_qoi_len = data_shape.quest_qoi_k[1]

    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = build_bool_sequence(demo_num, mode, shot_num_min)
    cond_len_list_raw = [demo_cond_len] * demo_num + [quest_cond_len]
    qoi_kv_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
    qoi_k_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
    cond_len_list = [i * j for i, j in zip(cond_bool_list, cond_len_list_raw)]
    qoi_kv_len_list = [i * j for i, j in zip(qoi_kv_bool_list, qoi_kv_len_list_raw)]
    qoi_k_len_list = [i * j for i, j in zip(qoi_k_bool_list, qoi_k_len_list_raw)]

    basic_mask = build_basic_mask(cond_len_list=cond_len_list,
                                  qoi_kv_len_list=qoi_kv_len_list,
                                  qoi_k_len_list=qoi_k_len_list,
                                  compact=compact)
    if config['index_mode'] == 'learn':
      index_matrix = build_index_integer(cond_len_list=cond_len_list,
                                          qoi_kv_len_list=qoi_kv_len_list,
                                          qoi_k_len_list=qoi_k_len_list)
    else:
      raise ValueError('index_mode {} not supported'.format(config['index_mode']))
    
    out_mask = build_out_mask(cond_len_list=cond_len_list,
                              qoi_kv_len_list=qoi_kv_len_list,
                              qoi_k_len_list=qoi_k_len_list,
                              num_range=(shot_num_min, demo_num + 1))

    # add prefix
    if caption_len > 0:
        basic_mask = torch.cat((torch.zeros(caption_len, basic_mask.size(1), dtype=torch.bool), basic_mask), dim=0)
        caption_mask = torch.ones(basic_mask.size(0), caption_len)
        if config['causal'] == 'caption':   
            caption_mask = torch.tril(caption_mask)             
        basic_mask = torch.cat((caption_mask, basic_mask), dim=1)
        out_mask = torch.cat((torch.zeros(caption_len, dtype=torch.bool), out_mask), dim=0)
    
    if config['causal'] == 'all':
        basic_mask = torch.tril(basic_mask)

    if return_shape_list:
        return basic_mask, index_matrix, out_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list
    else:
        return basic_mask, index_matrix, out_mask


class ICONGPT2(nn.Module):
  def __init__(self, config, data_shape,
               model_name='gpt2', pretrained=True):
    super(ICONGPT2, self).__init__()
    
    # Load the pre-trained GPT-2 model and tokenizer
    self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    gpt_config = GPT2Config.from_pretrained(model_name)
    gpt_config.resid_pdrop = 0
    gpt_config.attn_pdrop = 0
    gpt_config.embd_pdrop = 0
    gpt_config.summary_first_dropout = 0
    if pretrained:
      self.gpt2 = GPT2Model.from_pretrained(model_name, config=gpt_config) # without LM Head
      print(self.gpt2.config)
      print('='*50, 'pretrained model loaded', '='*50)
    else:
      self.gpt2 = GPT2Model(config = gpt_config) # without LM Head
      print(self.gpt2.config)    
      print('='*50, 'random initialized model loaded', '='*50)

    self.config = config

    if config['index_mode'] == 'learn':
      input_dim = config['k_dim'] + config['v_dim']
      self.func_pos_embedding = nn.Embedding((self.config['demo_max_num']) * 3, self.gpt2.config.n_embd)
    
    output_dim = config['v_dim']

    if self.config['input_net'] == 'linear':
      self.input_layer = nn.Linear(input_dim, self.gpt2.config.n_embd)
    else: # mlp with hidden_dim, gelu
      self.input_layer = nn.Sequential(
          nn.Linear(input_dim, self.config['input_net']['hidden_dim']),
          nn.GELU(),
          nn.Linear(self.config['input_net']['hidden_dim'], self.gpt2.config.n_embd)
      )
    
    if self.config['output_net'] == 'linear':
      self.output_layer = nn.Linear(self.gpt2.config.n_embd, output_dim)
    else:
      self.output_layer = nn.Sequential(
          nn.Linear(self.gpt2.config.n_embd, self.config['output_net']['hidden_dim']),
          nn.GELU(),
          nn.Linear(self.config['output_net']['hidden_dim'], output_dim)
      )
    
    basic_mask_with_caption, index_matrix_with_caption, out_mask_with_caption = build_matrices_from_data_shape(data_shape, config, compact = True, 
                                                                                                mode = 'train', caption_len = config['caption_len'], shot_num_min = 0)
    basic_mask_without_caption, index_matrix_without_caption, out_mask_without_caption = build_matrices_from_data_shape(data_shape, config, compact = True, 
                                                                                           mode = 'train', caption_len = 0, shot_num_min = 1)


    print("basic_mask_with_caption shape: ", basic_mask_with_caption.shape)
    print("index_matrix_with_caption shape: ", index_matrix_with_caption.shape)
    print("out_mask_with_caption shape: ", out_mask_with_caption.shape)
    print("basic_mask_without_caption shape: ", basic_mask_without_caption.shape)
    print("index_matrix_without_caption shape: ", index_matrix_without_caption.shape)
    print("out_mask_without_caption shape: ", out_mask_without_caption.shape)

    self.register_buffer('basic_mask_with_caption', basic_mask_with_caption)
    self.register_buffer('index_matrix_with_caption', index_matrix_with_caption)
    self.register_buffer('out_mask_with_caption', out_mask_with_caption)
    self.register_buffer('basic_mask_without_caption', basic_mask_without_caption)
    self.register_buffer('index_matrix_without_caption', index_matrix_without_caption)
    self.register_buffer('out_mask_without_caption', out_mask_without_caption)

    position_ids = torch.arange(0, self.config['caption_len'], dtype=torch.long).unsqueeze(0) # [1, caption_len]
    self.register_buffer('position_ids', position_ids)

    print('='*50, 'model created', '='*50)

  def basic_forward(self, data, mode, index_matrix, basic_mask, position_ids, caption_len, shot_num_min):
    '''
    mask shape: batch_size, seq_length, seq_length
    will replace the causal mask in the original model if not None
    assume index_matrix and basic_mask are already on device, without batch_size
    '''
    demo_num = data.demo_cond_k.shape[1]
    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = build_bool_sequence(demo_num, mode = mode, shot_num_min = shot_num_min)

    # build sequence
    data_sequence = build_data_sequence_batch(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list) # [batch_size, seq_length, dim]
    if self.config['index_mode'] == 'learn':
      index_matrix_batch = self.func_pos_embedding(index_matrix)
      index_matrix_batch = index_matrix_batch.unsqueeze(0).repeat(data_sequence.shape[0], 1, 1) # [batch_size, seq_length, input_dim]
      data_emb = self.input_layer(data_sequence) + index_matrix_batch # [batch_size, seq_length, n_embd]
    else:
      raise ValueError('index_mode {} not supported'.format(self.config['index_mode']))
    
    if caption_len > 0:
      position_emb = self.gpt2.wpe(position_ids) # [1, caption_len, n_embd]
      caption_emd = self.gpt2.wte(data.input_id) + position_emb # [batch_size, caption_len, n_embd]
      total_emb = torch.cat((caption_emd, data_emb), dim = 1) # [batch_size, seq_length + caption_len, n_embd]
    else:
      total_emb = data_emb
  
    # build mask
    data_mask = build_data_mask_batch(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)# [batch_size, seq_length]
    if caption_len > 0:
      caption_mask = data.embedding_mask
      total_mask = torch.cat((caption_mask, data_mask), dim = 1) # [batch_size, seq_length + caption_len]
    else:
      total_mask = data_mask
    total_mask = total_mask.unsqueeze(1) * basic_mask.unsqueeze(0) # [batch_size, seq_length, seq_length]
    
    # forward
    hidden_state = self.gpt2(inputs_embeds = total_emb, attention_mask = total_mask)[0] # [batch_size, seq_length, n_embd]
    output = self.output_layer(hidden_state) # [batch_size, seq_length, output_dim]
    return output

  def forward_without_caption(self, data):
    out = self.basic_forward(data, 'train', self.index_matrix_without_caption, self.basic_mask_without_caption,
                            position_ids = None,  caption_len = 0, shot_num_min = 1)
    out = out[:, self.out_mask_without_caption, :]
    return out
  
  def forward_with_caption(self, data):
    out = self.basic_forward(data, 'train', self.index_matrix_with_caption, self.basic_mask_with_caption,
                            position_ids = self.position_ids, caption_len = self.config['caption_len'], shot_num_min = 0)
    out = out[:, self.out_mask_with_caption, :]
    return out
  
  def predict(self, data, caption_len):
    shot_num_min = data.demo_cond_k.shape[1] # assume all demos are used
    data_shape = inputdata_transform(lambda x: x[0,...].shape, data)
    basic_mask, index_matrix, out_mask = build_matrices_from_data_shape(data_shape, self.config, compact = True, 
                                                                    mode = 'test', caption_len = caption_len, shot_num_min = shot_num_min)
    
    input_device = data.demo_cond_k.device
    basic_mask = basic_mask.to(input_device)
    index_matrix = index_matrix.to(input_device)
    
    if caption_len > 0:
      position_ids = torch.arange(0, data.input_id.size(1), dtype=torch.long).unsqueeze(0).expand(data.input_id.size(0), -1) # [batch_size, caption_len]
      position_ids = position_ids.to(data.input_id.device)
    else:
      position_ids = None
    
    out = self.basic_forward(data, 'test', index_matrix, basic_mask, position_ids,
                             caption_len = caption_len, shot_num_min = shot_num_min)
    out = out[:,-data.quest_qoi_mask.shape[-1]:,:] # [batch_size, seq_length, output_dim]
    return out

  def build_input_data(self, input_id, embedding_mask,
                  demo_cond_k, demo_cond_v, demo_cond_mask,
                  demo_qoi_k, demo_qoi_v, demo_qoi_mask,
                  quest_cond_k, quest_cond_v, quest_cond_mask,
                  quest_qoi_k, quest_qoi_mask):
      return InputData(input_id = input_id, embedding_mask = embedding_mask,
                          demo_cond_k = demo_cond_k, demo_cond_v = demo_cond_v, demo_cond_mask = demo_cond_mask,
                          demo_qoi_k = demo_qoi_k, demo_qoi_v = demo_qoi_v, demo_qoi_mask = demo_qoi_mask,
                          quest_cond_k = quest_cond_k, quest_cond_v = quest_cond_v, quest_cond_mask = quest_cond_mask,
                          quest_qoi_k = quest_qoi_k, quest_qoi_mask = quest_qoi_mask)


  def forward(self, input_id, embedding_mask,
                  demo_cond_k, demo_cond_v, demo_cond_mask,
                  demo_qoi_k, demo_qoi_v, demo_qoi_mask,
                  quest_cond_k, quest_cond_v, quest_cond_mask,
                  quest_qoi_k, quest_qoi_mask,
                  mode):
    data = self.build_input_data(input_id, embedding_mask,
                                demo_cond_k, demo_cond_v, demo_cond_mask,
                                demo_qoi_k, demo_qoi_v, demo_qoi_mask,
                                quest_cond_k, quest_cond_v, quest_cond_mask,
                                quest_qoi_k, quest_qoi_mask)
    if mode == "forward_with_caption":
      return self.forward_with_caption(data)
    elif mode == "forward_without_caption":
      return self.forward_without_caption(data)
    elif type(mode) == int:
      return self.predict(data, caption_len = mode)
    else:
      raise ValueError("mode {} not supported".format(mode))
    

def test():
  import dataloader
  import numpy as np
  import utils
  from pprint import pprint
  from models_utils import plot_model_consts

  data_config = utils.load_json('config_data/train_lm_config.json')
  # data_config['load_list'] = []
  data_provider = dataloader.DataProvider(seed = 1, config = data_config, 
                              file_names = '/home/shared/icon/data/data0910c/train*',
                              batch_size = 2, shuffle_buffer_size = 10,
                              num_devices = 0, real_time = True)
  
  equation, caption, data, label = data_provider.get_next_data()
  dataloader.print_eqn_caption(equation, caption, decode = False)
  pprint(inputdata_transform(lambda x: (type(x), x.shape), data)) 

  data_shape = inputdata_transform(lambda x: x[0,...].shape, data)
  model_config = utils.load_json('config_model/model_gpt2_config.json')
  model_config['index_mode'] = 'learn'
  model_config['caption_len'] = 300
  model_config['causal'] = 'caption'
  model = ICONGPT2(model_config, data_shape, model_name = 'gpt2', pretrained = True)
  model.to(device)
  print(model)

  data = inputdata_transform(lambda x: torch.tensor(x).to(device), data)
  out = model.forward_without_caption(data)
  print(out.shape)

  out = model.forward_with_caption(data)
  print(out.shape)

  out = model.predict(data, caption_len = 0)
  print(out.shape)

  out = model.predict(data, caption_len = model_config['caption_len'])
  print(out.shape)
  

if __name__ == "__main__":
  test()