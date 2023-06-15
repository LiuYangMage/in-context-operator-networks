import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import utils


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  '''add a LayerNorm layer and apply to x'''
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)


@dataclasses.dataclass
class SelfAttnTransformer(hk.Module):
  '''
  self-attention transformer
  '''

  num_heads: int
  num_layers: int
  model_size: int # the size of embedding or h
  QK_size: int
  V_size: int
  widening_factor: int = 4
  initializer: str = 'glorot_uniform'
  name: Optional[str] = None

  def __call__(
      self,
      embeddings: jnp.ndarray,  # [...,T, model_size]
      mask = None,  # [...,1, T, T]
  ) -> jnp.ndarray:  # [..., T, model_size]

    if self.initializer == 'glorot_uniform':
      initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform") # glorot_uniform
    elif self.initializer == 'layer_scale':
      initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    else:
      raise NotImplementedError

    e_norm = layer_norm(embeddings)
    h = e_norm
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.QK_size,
          value_size=self.V_size,
          model_size=self.model_size,
          w_init=initializer)
      h_attn = attn_block(h, h, h, mask=mask)
      h = h + h_attn
      h = layer_norm(h)

      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * self.model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(self.model_size, w_init=initializer),
      ])
      h_dense = dense_block(h)
      h = h + h_dense
      h = layer_norm(h)

    return h


@dataclasses.dataclass
class CrossAttnTransformer(hk.Module):
  '''
  cross-attention transformer
  Note that in MultiHeadAttention, the query, key, value will be reshaped via linear projection
  query -> Q [t, key_size]
  key   -> K [T, key_size]
  value -> V [T, value_size]
  '''

  num_heads: int
  num_layers: int
  model_size: int # dim for query
  QK_size: int # dim for Q and K
  V_size: int # dim for V
  widening_factor: int = 4
  initializer: str = 'glorot_uniform'
  name: Optional[str] = None


  def __call__(
      self,
      query: jnp.ndarray,  # [t, model_size]
      key: jnp.ndarray,  # [T, key_size]
      value: jnp.ndarray, #[T, value_size]
      mask = None, #[1, t, T]
      final_norm = True
  ) -> jnp.ndarray:  # [t, D_o]

    if self.initializer == 'glorot_uniform':
      initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform") # glorot_uniform
    elif self.initializer == 'layer_scale':
      initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    else:
      raise NotImplementedError

    query_norm = layer_norm(query)
    key_norm = layer_norm(key)
    value_norm = layer_norm(value)

    for i in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.QK_size,
          w_init=initializer,
          value_size=self.V_size,
          model_size=self.model_size,
          name = "attn_{}".format(i))
                  
      this_query = attn_block(query = query_norm, key = key_norm, value = value_norm, mask = mask)
      query_norm = layer_norm(this_query + query_norm)
    
      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * self.model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(self.model_size, w_init=initializer),
          ], name = "dense_{}".format(i))
      
      this_query = dense_block(query_norm)
    
      if (i == self.num_layers) and not final_norm:
        query_norm = this_query + query_norm
      else:
        query_norm = layer_norm(this_query + query_norm)

    return query_norm


if __name__ == "__main__":

  query_size = 3
  key_size = 4
  value_size = 5
  QK_size = 10
  V_size = 12

  t = 20
  T = 40
  query = jax.random.normal(jax.random.PRNGKey(1), [t,query_size])
  key = jax.random.normal(jax.random.PRNGKey(2), [T,key_size])
  value = jax.random.normal(jax.random.PRNGKey(3), [T,value_size])

  def f(q, k ,v):
    net = CrossAttnTransformer(num_heads = 8,
                               num_layers = 4,
                               model_size = query_size,
                               QK_size = QK_size,
                               V_size = V_size,
                               widening_factor = 4)
    return net(q,k,v)

  f = hk.transform(f) 
  rng_key = jax.random.PRNGKey(1234)
  params = f.init(rng_key, query, key, value)
  out_query = f.apply(params, rng_key, query, key, value)
  utils.print_pytree(params)
  print(out_query.shape) # (20,3)
