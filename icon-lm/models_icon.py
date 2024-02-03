import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import haiku as hk
import utils
from transformer_hk import SelfAttnTransformer, CrossAttnTransformer
from einshape import jax_einshape as einshape
import dataclasses
import jax.tree_util as tree

def build_index_fn(demo_max_num, demo_num, demo_cond_len, demo_qoi_len, quest_cond_len):
  '''
  @params:
    demo_num: int, number of demonstrations
    demo_cond_len: int, length of demonstration condition
    demo_qoi_len: int, length of demonstration qoi
    quest_cond_len: int, length of query condition
    quest_qoi_len: int, length of query qoi
  @return:
    index: 2D array, [demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len, demo_max_num + 1]
  '''
  index_list = []
  for i in range(demo_num):
    cond_index = jax.nn.one_hot(jnp.ones((demo_cond_len,)) * i, demo_max_num + 1)
    qoi_index = - jax.nn.one_hot(jnp.ones((demo_qoi_len,)) * i, demo_max_num + 1)
    index_list.append(cond_index)
    index_list.append(qoi_index)
  cond_index = jax.nn.one_hot(jnp.ones((quest_cond_len,)) * demo_max_num, demo_max_num + 1)
  index_list.append(cond_index)
  index = jnp.concatenate(index_list, axis = 0)
  return index

@partial(jax.jit, static_argnums = (1,))
def build_prompt_mask_query_fn(data, demo_max_num):
  '''no batch, no device'''
  demo_cond_kv = jnp.concatenate([data.demo_cond_k, data.demo_cond_v], axis = -1) # (demo_num, demo_cond_len, k_dim + v_dim)
  demo_qoi_kv = jnp.concatenate([data.demo_qoi_k, data.demo_qoi_v], axis = -1) # (demo_num, demo_qoi_len, k_dim + v_dim)
  demo_kv = jnp.concatenate([demo_cond_kv, demo_qoi_kv], axis = -2) # (demo_num, demo_cond_len + demo_qoi_len, k_dim + v_dim)
  demo_kv = einshape("...ijk->...(ij)k", demo_kv) # (demo_num * (demo_cond_len + demo_qoi_len), k_dim + v_dim)

  # quest_num = 1
  quest_cond_kv = jnp.concatenate([data.quest_cond_k, data.quest_cond_v], axis = -1) # (1, quest_cond_len, k_dim + v_dim)
  quest_kv = einshape("...ijk->...(ij)k", quest_cond_kv) # (quest_cond_len, k_dim + v_dim)
  
  prompt_kv = jnp.concatenate([demo_kv, quest_kv], axis = -2) # (demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len, k_dim + v_dim)
  
  demo_num = data.demo_cond_k.shape[-3]
  demo_cond_len = data.demo_cond_k.shape[-2]
  demo_qoi_len = data.demo_qoi_k.shape[-2]
  quest_cond_len = data.quest_cond_k.shape[-2]
  index = build_index_fn(demo_max_num, demo_num, demo_cond_len, demo_qoi_len, quest_cond_len) # (demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len, demo_max_num + 1)
  prompt = jnp.concatenate([prompt_kv, index], axis = -1) # (demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len, k_dim + v_dim + demo_max_num + 1)
  
  demo_mask = jnp.concatenate([data.demo_cond_mask, data.demo_qoi_mask], axis = -1) # (demo_num, demo_cond_len + demo_qoi_len)
  demo_mask = einshape("...ij->...(ij)", demo_mask) # (demo_num * (demo_cond_len + demo_qoi_len),)
  quest_mask = einshape("...ij->...(ij)", data.quest_cond_mask) # (quest_cond_len,)
  mask = jnp.concatenate([demo_mask, quest_mask], axis = -1) # (demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len,)

  prompt = prompt * mask[..., None]
  query = data.quest_qoi_k[0,...] # (quest_qoi_len, k_dim)

  return (prompt, # 2D (demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len, k_dim + v_dim + demo_max_num + 1)
          mask, # 1D (demo_num * (demo_cond_len + demo_qoi_len) + quest_cond_len,)
          query, # 2D (quest_qoi_len, k_dim)
          )

build_prompt_mask_query_batch_fn = jax.jit(jax.vmap(build_prompt_mask_query_fn, in_axes = [0, None]), static_argnums = (1,))
build_prompt_mask_query_pmap_batch_fn = jax.jit(jax.vmap(build_prompt_mask_query_batch_fn, in_axes = [0, None]), static_argnums = (1,))

@dataclasses.dataclass
class SolverModel(hk.Module):
  '''
  kv_projection: preprocess linear projection, project demonstrations and cond_kv to model_size
  q_projection: preprocess linear projection, project query to q_size
  encoder: self-attention transformer. qkv: demonstrations, cond_kv; output: the embedding of the system
  decoder: cross-attention transformer. kv: the embedding of the system; q: query; output: embedding of qoi_v
  out_projection: postprocess linear projection, project the output of decoder to qoi_v
  '''
  def __init__(self, config):
    '''
    @params:
      q_size: dim for query after preprocess projection
      model_size: dim for key and value after preprocess projection
      out_size: dim for qoi_v after postprocess projection
      QK_size: dim for Q and K in self-attn
      V_size: dim for V in attn
    '''
    super(SolverModel, self).__init__()
    self.config = config
    self.kv_projection = hk.Linear(config['encoder']['model_size'])
    self.q_projection = hk.Linear(config['decoder']['model_size'])

    self.encoder = SelfAttnTransformer(**config['encoder'])
    self.decoder = CrossAttnTransformer(**config['decoder'])

    self.out_projection = hk.Linear(config['out_size'])

  def __call__(self, data):
    '''
    @params:
      data
    @return:
      qoi_v: 2D array, [quest_qoi_len, out_size]
    '''
    prompt, mask, query = build_prompt_mask_query_fn(data, self.config['demo_max_num'])
    kv_embedding = self.kv_projection(prompt)
    sys_embedding = self.encoder(kv_embedding, einshape("i->kji", mask, k=1, j=len(mask))) # (prompt_len + caption_emb_len, model_size)
    q_embedding = self.q_projection(query)
    out_embedding = self.decoder(query = q_embedding, key = sys_embedding, 
                                value = sys_embedding, mask = einshape("i->kji", mask, k=1, j=len(query)))
    qoi_v = self.out_projection(out_embedding)
    return qoi_v


def build_network_fn(data, key, config):
  def f(data):
    net = SolverModel(config)
    return net(data)

  f = hk.transform(f)
  data = tree.tree_map(lambda x: x[0,0], data) # take off device and batch dimension
  params = f.init(key, data)
  return jax.jit(f.apply), params


if __name__ == "__main__":
  from jax.config import config
  config.update('jax_enable_x64', True)
  from dataloader import DataProvider, print_eqn_caption
  from pprint import pprint
  import jax.tree_util as tree

  np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
  config = utils.load_json('train_config.json')
  data_provider = DataProvider(seed = 1, config = config,
                              file_names = '../data/data0614a/train*',
                              batch_size = 16, shuffle_buffer_size = 1000,)
  equation, caption, data, label =  data_provider.get_next_data()
  print_eqn_caption(equation, caption)
  print(tree.tree_map(lambda x: x.shape, data)) 

  index = build_index_fn(demo_max_num = config['demo_num'] + 2, 
                         demo_num = config['demo_num'], 
                         demo_cond_len = config['demo_cond_len'],
                         demo_qoi_len = config['demo_qoi_len'],
                         quest_cond_len = config['quest_cond_len'])
  print(index)
  print('============================')
  
  prompt, mask, query = build_prompt_mask_query_pmap_batch_fn(data, 5)
  print(prompt[0,0,...])
  
  print("index", index.shape)
  print('prompt', prompt.shape)
  print('mask', mask.shape)
  print('query', query.shape)

