"""
neat implementation of transformers in flax
adapted from here (changed significantly):
https://github.com/satojkovic/vit-jax-flax/blob/main/vit_jax_flax/vit.py

"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.tree_util as tree
from typing import Optional
from attention_module import MultiHeadDotProductAttention as MHAttention
from typing import (Any, Callable, Tuple, Optional)

from flax.linen.attention import dot_product_attention

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


init_dict = {'lecun_normal': nn.initializers.lecun_normal(),
            'glorot_uniform': nn.initializers.glorot_uniform(),
            'glorot_normal': nn.initializers.glorot_normal(),
}


attn_fn_dict = {'vanilla': dot_product_attention,
                }


def translate_config(config):
  new_config = dict(config)
  if 'kernel_init' in new_config:
    new_config['kernel_init'] = init_dict[new_config['kernel_init']]
  if 'attention_fn' in new_config:
    new_config['attention_fn'] = attn_fn_dict[new_config['attention_fn']]
  return new_config


class MLP(nn.Module):
  hidden_dim: int
  out_dim: int
  dropout_rate: Optional[float]
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]
  depth: int

  @nn.compact
  def __call__(self, inputs, train=True):
    x = inputs
    for i in range(self.depth):
      x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
      x = nn.gelu(x)
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = nn.Dense(features=self.out_dim, kernel_init=self.kernel_init)(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    return x

class ResNetMLP(nn.Module):
  hidden_dim: int
  out_dim: int
  dropout_rate: Optional[float]
  kernel_init: Callable
  depth: int

  @nn.compact
  def __call__(self, x, train=True):
    x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name='input_layer')(x)
    for i in range(self.depth):
      initial_x = x
      x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name=f'block_{i}_dense_1')(x)
      x = nn.gelu(x)
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
      x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init, name=f'block_{i}_dense_2')(x)
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
      x = x + initial_x  # Residual connection
    x = nn.Dense(self.out_dim, kernel_init=self.kernel_init, name='output_layer')(x)
    return x


class SelfAttnTransformer(nn.Module):
  n_layers: int
  n_heads: int
  head_dim: int
  model_dim: int # the dimension of the input/output of the attention block and dense block
  dropout_rate: Optional[float]
  widening_factor: int # widening factor for hidden_dim
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention

  @nn.compact
  def __call__(self, inputs, mask = None):
    # Attention Block
    x = inputs
    for _ in range(self.n_layers):
      attn_block = MHAttention(num_heads = self.n_heads, 
                              qkv_features = self.head_dim * self.n_heads, 
                              out_features = self.model_dim,
                              dropout_rate = self.dropout_rate,
                              kernel_init = self.kernel_init,
                              attention_fn = self.attention_fn)
      x = attn_block(inputs_q = x, inputs_k = x, inputs_v = x, mask = mask) + x
      x = nn.LayerNorm()(x)
      dense_block = MLP(hidden_dim = self.model_dim * self.widening_factor, 
                        dropout_rate = self.dropout_rate, 
                        out_dim = self.model_dim, 
                        kernel_init = self.kernel_init,
                        depth = 1)
      x = dense_block(x) + x
      x = nn.LayerNorm()(x)
    
    return x


class CrossAttnTransformer(nn.Module):
  """
  Cross Attention Transformer without self attention layer
  """
  n_layers: int
  n_heads: int
  head_dim: int
  model_dim: int # the dimension of the input/output of the attention block and dense block
  dropout_rate: Optional[float]
  widening_factor: int # widening factor for hidden_dim
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention

  @nn.compact
  def __call__(self, inputs_q, inputs_k, inputs_v, mask = None):
    # Attention Block
    x = inputs_q
    for _ in range(self.n_layers):
      attn_block = MHAttention(num_heads = self.n_heads, 
                              qkv_features = self.head_dim * self.n_heads, 
                              out_features = self.model_dim,
                              dropout_rate = self.dropout_rate,
                              kernel_init = self.kernel_init,
                              attention_fn = self.attention_fn)
      y = attn_block(inputs_q = x, inputs_k = inputs_k, inputs_v = inputs_v, mask = mask)
      assert y.shape == x.shape
      x = x + y
      x = nn.LayerNorm()(x)
      dense_block = MLP(hidden_dim = self.model_dim * self.widening_factor, 
                        dropout_rate = self.dropout_rate, 
                        out_dim = self.model_dim, 
                        kernel_init = self.kernel_init,
                        depth = 1)
      y = dense_block(x)
      assert y.shape == x.shape
      x = x + y
      x = nn.LayerNorm()(x)
    
    return x


  
def test():
  # Initialize random PRNGKey
  key = jax.random.PRNGKey(42)
  rngs = {'params': jax.random.PRNGKey(20), 'dropout': jax.random.PRNGKey(10)}

  # Define input shape and model parameters
  input_q_length = 10
  input_kv_length = 20
  n_layers = 4
  n_heads = 8
  head_dim = 64
  model_dim = 256
  dropout_rate = 0.0
  widening_factor = 4
  bs = 1

  # Initialize model instances
  self_attn_transformer = SelfAttnTransformer(
      n_layers=n_layers,
      n_heads=n_heads,
      head_dim=head_dim,
      model_dim=model_dim,
      dropout_rate=dropout_rate,
      widening_factor=widening_factor,
      kernel_init=init_dict['glorot_uniform'],
  )

  cross_attn_transformer = CrossAttnTransformer(
      n_layers=n_layers,
      n_heads=n_heads,
      head_dim=head_dim,
      model_dim=model_dim,
      dropout_rate=dropout_rate,
      widening_factor=widening_factor,
      kernel_init=init_dict['glorot_uniform'],
  )

  # Test self-attention transformer
  inputs = jax.random.normal(key, (bs, input_q_length, model_dim))
  mask = jnp.ones((bs, input_q_length,))
  print(self_attn_transformer.tabulate(rngs = rngs, inputs = inputs, mask = mask))

  # Test cross-attention transformer
  inputs_q = jax.random.normal(key, (bs, input_q_length, model_dim))
  inputs_k = jax.random.normal(key, (bs, input_kv_length, model_dim * 2))
  inputs_v = jax.random.normal(key, (bs, input_kv_length, model_dim * 3))
  mask = jnp.ones((bs, input_kv_length,))
  print(cross_attn_transformer.tabulate(rngs, inputs_q, inputs_k, inputs_v, mask))


if __name__ == "__main__":
  test()
