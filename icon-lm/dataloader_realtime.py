import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')
import os
from utils import load_json
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def split_equation(equation, lead_n=4):
    '''equation is a string, e.g. 'ode_auto_linear1_forward_0.806_0.047'''
    # Split the string by the separator '_'
    split_list = equation.split('_')
    # Extract the first four elements for the identifier
    identifier = '_'.join(split_list[:lead_n])
    # Convert the rest to float
    params = [float(param) for param in split_list[lead_n:]]
    return identifier, params

# define a parser function to parse the serialized example
def parse_function(example_proto, config):
    '''
    @return
      equation: string describing the equation
      caption: caption strings (n,)
      embedding_raw:  embedding of the caption strings, (n, len, embedding_dim)
      embedding_pool: pooled embedding of the caption strings, (n, embedding_dim)
      embedding_mask: mask of the caption strings, (n, len)
      cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
      cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
      qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
      qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
    '''
    feature_description = {
        'equation': tf.io.FixedLenFeature([], tf.string),
        'cond_k': tf.io.FixedLenFeature([], tf.string),
        'cond_v': tf.io.FixedLenFeature([], tf.string),
        'qoi_k': tf.io.FixedLenFeature([], tf.string),
        'qoi_v': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    equation = parsed_example['equation']
    cond_k = tf.io.parse_tensor(parsed_example['cond_k'], out_type=tf.float32)
    cond_v = tf.io.parse_tensor(parsed_example['cond_v'], out_type=tf.float32)
    qoi_k = tf.io.parse_tensor(parsed_example['qoi_k'], out_type=tf.float32)
    qoi_v = tf.io.parse_tensor(parsed_example['qoi_v'], out_type=tf.float32)

    return equation, cond_k, cond_v, qoi_k, qoi_v


def select_caption(equation, cond_k, cond_v, qoi_k, qoi_v, config):
    # make dummy captions
    caption = tf.zeros((), dtype = tf.string)
    embedding_raw = tf.zeros((), dtype = tf.float32)
    embedding_pool = tf.zeros((), dtype = tf.float32)
    embedding_mask = tf.zeros((), dtype = tf.bool)
    input_id = tf.zeros((), dtype = tf.int32)

    return equation, caption, input_id, embedding_raw, embedding_pool, embedding_mask, cond_k, cond_v, qoi_k, qoi_v
