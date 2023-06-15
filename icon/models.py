import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import haiku as hk
import utils
from transformer import SelfAttnTransformer, CrossAttnTransformer
from einshape import jax_einshape as einshape
import dataclasses



def pad_and_concat(demos, quest_cond, k_dim, v_dim, cond_len, qoi_len, demo_num):
  '''
  pad and concatenate the inputs (demos and condition), used in data loader
  @params:
    demos: a list of demos (demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v)
    quest_cond: question condition (quest_cond_k, quest_cond_v)
    k_dim: max dim for k in prompt
    v_dim: max dim for v in prompt
    cond_len: max length of each demo cond
    qoi_len: max length of each demo qoi
    demo_num: max num of demos
  @return:
    prompt: 2D array as the model input
  '''
  # demo_num + 1 sequences, 1 stands for cond
  prompt = np.zeros((demo_num * (cond_len + qoi_len) + cond_len, k_dim + v_dim + demo_num + 1))
  mask = np.zeros((demo_num * (cond_len + qoi_len) + cond_len,))

  for i, (demo_cond_k, demo_cond_v, demo_qoi_k, demo_qoi_v) in enumerate(demos):

    demo_cond_len, demo_cond_k_dim = demo_cond_k.shape
    demo_cond_len, demo_cond_v_dim = demo_cond_v.shape
    demo_qoi_len, demo_qoi_k_dim = demo_qoi_k.shape
    demo_qoi_len, demo_qoi_v_size = demo_qoi_v.shape

    this_slice = prompt[i*(cond_len + qoi_len):i*(cond_len + qoi_len)+demo_cond_len,:]
    this_slice[:,0:demo_cond_k_dim] = demo_cond_k
    this_slice[:,k_dim:k_dim+demo_cond_v_dim] = demo_cond_v
    this_slice[:,k_dim+v_dim+i] = 1.0 #indicate demo index
    mask[i*(cond_len + qoi_len):i*(cond_len + qoi_len)+demo_cond_len] = 1.0

    this_slice = prompt[i*(cond_len + qoi_len)+cond_len:i*(cond_len + qoi_len)+cond_len+demo_qoi_len,:]
    this_slice[:,0:demo_qoi_k_dim] = demo_qoi_k
    this_slice[:,k_dim:k_dim+demo_qoi_v_size] = demo_qoi_v
    this_slice[:,k_dim+v_dim+i] = -1.0 #indicate demo index
    mask[i*(cond_len + qoi_len)+cond_len:i*(cond_len + qoi_len)+cond_len+demo_qoi_len] = 1.0

  quest_cond_k, quest_cond_v = quest_cond
  quest_cond_len, quest_cond_k_dim = quest_cond_k.shape
  quest_cond_len, quest_cond_v_dim = quest_cond_v.shape
  
  this_slice = prompt[demo_num * (cond_len + qoi_len):demo_num * (cond_len + qoi_len) + quest_cond_len, :]
  this_slice[:,0:quest_cond_k_dim] = quest_cond_k
  this_slice[:,k_dim:k_dim+quest_cond_v_dim] = quest_cond_v
  this_slice[:,k_dim+v_dim+demo_num] = 1.0 #indicate quest
  mask[demo_num * (cond_len + qoi_len):demo_num * (cond_len + qoi_len) + quest_cond_len] = 1.0

  return prompt, mask
    

@dataclasses.dataclass
class SolverModel(hk.Module):
  '''
  kv_projection: preprocess linear projection, project demonstrations and cond_kv to kv_size
  q_projection: preprocess linear projection, project query to q_size
  encoder: self-attention transformer. qkv: demonstrations, cond_kv; output: the embedding of the system
  decoder: cross-attention transformer. kv: the embedding of the system; q: query; output: embedding of qoi_v
  out_projection: postprocess linear projection, project the output of decoder to qoi_v
  '''
  def __init__(self, q_size: int,
                  kv_size: int,
                  qoi_v_size: int,
                  QK_size: int,
                  V_size: int,
                  num_heads: int,
                  num_layers: int,
                  initializer: str = 'glorot_uniform',
                  widening_factor: int = 4
                  ):
    '''
    @params:
      q_size: dim for query after preprocess projection
      kv_size: dim for key and value after preprocess projection
      qoi_v_size: dim for qoi_v after postprocess projection
      QK_size: dim for Q and K in self-attn
      V_size: dim for V in attn
    '''
    super(SolverModel, self).__init__()

    self.q_size = q_size
    self.kv_size = kv_size
    self.QK_size = QK_size
    self.V_size = V_size
    self.qoi_v_size = qoi_v_size

    self.kv_projection = hk.Linear(kv_size)
    self.q_projection = hk.Linear(q_size)

    self.encoder = SelfAttnTransformer(num_heads = num_heads,
                                       num_layers = num_layers,
                                       model_size = kv_size, 
                                       QK_size = QK_size, 
                                       V_size = V_size,
                                       initializer = initializer,
                                       widening_factor = widening_factor)
    self.decoder = CrossAttnTransformer(num_heads = num_heads,
                                        num_layers = num_layers,
                                        model_size = q_size,
                                        QK_size = QK_size,
                                        V_size = V_size,
                                        initializer = initializer,
                                        widening_factor = widening_factor)

    self.out_projection = hk.Linear(qoi_v_size)

  def __call__(self, prompt, mask, query):
    '''
    @params:
      prompt: 2D array, including demos and the condition
      mask: 1D, [len(prompt)]
      query: 2D array, query representing the key of qoi
    '''
    kv_embedding = self.kv_projection(prompt)
    sys_embedding = self.encoder(kv_embedding, einshape("i->kji", mask, k=1, j=len(prompt)))
    q_embedding = self.q_projection(query)
    out_embedding = self.decoder(query = q_embedding, key = sys_embedding, 
                                value = sys_embedding, mask = einshape("i->kji", mask, k=1, j=len(query)))
    qoi_v = self.out_projection(out_embedding)
    return qoi_v



if __name__ == "__main__":
  from jax.config import config
  config.update('jax_enable_x64', True)
  np.set_printoptions(threshold=np.inf)
  demos = []
  for i in range(4):
    demo_cond_k = np.ones((i+1,2))*0.1 + i
    demo_cond_v = np.ones((i+1,3))*0.2 + i
    demo_qoi_k = np.ones((i+2,3))*0.3 + i
    demo_qoi_v = np.ones((i+2,4))*0.4 + i
    demos.append((demo_cond_k,demo_cond_v, demo_qoi_k,demo_qoi_v))
  quest_cond = (np.ones((3,2))*10, np.ones((3,3))*10.1)
  prompt, mask = pad_and_concat(demos = demos, quest_cond = quest_cond, k_dim = 3, v_dim = 4, cond_len = 6, qoi_len = 6, demo_num = 5)
  print(prompt.shape, mask.shape)
  
  print(np.concatenate([prompt, 10*mask[:,None]], axis = 1))

  query = jax.random.uniform(jax.random.PRNGKey(1), (10,3))

  query_size = 3
  qoi_v_size = 5
  key_size = 4
  value_size = 5
  QK_size = 10
  V_size = 12

  def f(prompt, mask, query):
    net = SolverModel(q_size = 128,
                  kv_size = 128,
                  qoi_v_size = qoi_v_size,
                  QK_size = 128,
                  V_size = 128,
                  num_heads = 8,
                  num_layers = 4,
                  )
    return net(prompt, mask, query)

  f = hk.transform(f) 
  rng_key = jax.random.PRNGKey(1234)
  params = f.init(rng_key, prompt, mask, query)
  out = f.apply(params, rng_key, prompt, mask, query)
  utils.print_pytree(params)
  print(out.shape) # (10,5)
  assert out.shape == (len(query), qoi_v_size)

  key = jax.random.PRNGKey(42)
  # same random key, same permutation
  prompt_perm = jax.random.permutation(key, prompt, axis=0)
  mask_perm = jax.random.permutation(key, mask, axis=0)
  out_perm = f.apply(params, rng_key, prompt_perm, mask_perm, query)
  assert np.allclose(out, out_perm)


  prompt_long, mask_long = pad_and_concat(demos = demos, quest_cond = quest_cond, k_dim = 3, v_dim = 4, cond_len = 10, qoi_len = 10, demo_num = 5)
  print(prompt_long.shape, mask_long.shape)
  out_long = f.apply(params, rng_key, prompt_long, mask_long, query)
  assert np.allclose(out, out_long)
