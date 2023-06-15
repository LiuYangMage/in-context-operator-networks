import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from einshape import jax_einshape as einshape



tf.config.set_visible_devices([], device_type='GPU')

def serialize_element(equation, cond_k, cond_v, qoi_k, qoi_v):
    '''
    equation: string describing the equation
    cond_k: condition key, 3D, (num, cond_length, cond_k_dim)
    cond_v: condition value, 3D, (num, cond_length, cond_v_dim)
    qoi_k: qoi key, 3D, (num, qoi_length, qoi_k_dim)
    qoi_v: qoi value, 3D, (num, qoi_length, qoi_v_dim)
    '''
    cond_k = cond_k.astype(np.float32)
    cond_v = cond_v.astype(np.float32)
    qoi_k = qoi_k.astype(np.float32)
    qoi_v = qoi_v.astype(np.float32)
    
    feature = {
      'equation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[equation.encode("utf-8")])),
      'cond_k': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cond_k).numpy()])),
      'cond_v': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(cond_v).numpy()])),
      'qoi_k': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(qoi_k).numpy()])),
      'qoi_v': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(qoi_v).numpy()])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



def write_ode_tfrecord(name, eqn_type, all_params, all_ts, all_cs, all_us, problem_types):
  num = all_us[0].shape[0]
  if "forward" in problem_types:
    filename = "{}_{}_forward.tfrecord".format(name, eqn_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_expand, control, traj in zip(all_params, all_ts, all_cs, all_us):
        equation_name = "{}_forward_{}".format(eqn_type, params)
        cond_k_c = jnp.pad(ts_expand[:,:-1,:], ((0,0),(0,0),(0,1)), mode = 'constant', constant_values = 0.0)
        cond_k_i = einshape('i->jki', jnp.array([0.0,1.0]), j = num, k = 1)
        cond_k = jnp.concatenate([cond_k_c, cond_k_i], axis = 1)
        cond_v_c = control[:,:-1,:]
        cond_v_i = traj[:,0:1,:]
        cond_v = jnp.concatenate([cond_v_c, cond_v_i], axis = 1)
        if not printed:
          print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                    equation_name, cond_k.shape, cond_v.shape, ts_expand.shape, traj.shape), flush=True)
          printed = True
        if np.sum(traj) != np.nan:
          s_element= serialize_element(equation = equation_name, cond_k = cond_k, cond_v = cond_v, qoi_k = ts_expand, qoi_v = traj)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if "inverse" in problem_types:
    filename = "{}_{}_inverse.tfrecord".format(name, eqn_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_expand, control, traj in zip(all_params, all_ts, all_cs, all_us):
        equation_name = "{}_inverse_{}".format(eqn_type, params)
        if not printed:
          print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                    equation_name, ts_expand.shape, traj.shape, ts_expand.shape, control.shape), flush=True)
          printed = True
        if np.sum(traj) != np.nan:
          s_element= serialize_element(equation = equation_name, cond_k = ts_expand, cond_v = traj, qoi_k = ts_expand, qoi_v = control)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

def write_series_tfrecord(name, eqn_type,all_params, all_ts_first, all_us_first, all_ts_second, all_us_second, problem_types):
    
  if "forward" in problem_types:
    filename = "{}_{}_forward.tfrecord".format(name, eqn_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_first, us_first, ts_second, us_second in zip(all_params, all_ts_first, all_us_first, all_ts_second, all_us_second):
        equation_name = "{}_forward_{}".format(eqn_type, params)
        if not printed:
          print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                equation_name, ts_first.shape, us_first.shape, ts_second.shape, us_second.shape), flush=True)
          printed = True
        if jnp.sum(us_second) != jnp.nan:
          s_element= serialize_element(equation = equation_name, cond_k = ts_first, cond_v = us_first, qoi_k = ts_second, qoi_v = us_second)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")
  
  if "inverse" in problem_types:
    filename = "{}_{}_inverse.tfrecord".format(name, eqn_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, ts_first, us_first, ts_second, us_second in zip(all_params, all_ts_first, all_us_first, all_ts_second, all_us_second):
        equation_name = "{}_inverse_{}".format(eqn_type, params)
        if not printed:
          print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                equation_name, ts_second.shape, us_second.shape, ts_first.shape, us_first.shape), flush=True)
          printed = True
        if jnp.sum(us_second) != jnp.nan:
          s_element= serialize_element(equation = equation_name, cond_k = ts_second, cond_v = us_second, qoi_k = ts_first, qoi_v = us_first)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

def write_pde_tfrecord(name, eqn_type, all_params, all_xs, all_ks, all_us, problem_types):
  if "forward" in problem_types:
    filename = "{}_{}_forward.tfrecord".format(name, eqn_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, xs, ks, us in zip(all_params, all_xs, all_ks, all_us):
        if np.sum(us) != np.nan:
          equation_name = "{}_forward_{}".format(eqn_type, params)
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, xs.shape, ks.shape, xs.shape, us.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = xs, cond_v = ks, qoi_k = xs, qoi_v = us)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if "inverse" in problem_types:
    filename = "{}_{}_inverse.tfrecord".format(name, eqn_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, xs, ks, us in zip(all_params, all_xs, all_ks, all_us):
        if np.sum(us) != np.nan:
          equation_name = "{}_inverse_{}".format(eqn_type, params)
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, xs.shape, us.shape, xs.shape, ks.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = xs, cond_v = us, qoi_k = xs, qoi_v = ks)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")


def write_mfc_gparam_solve_tfrecord(name, eqn_type, all_params, all_xs, all_rhos, problem_types):
  '''
  rhos: [num, xlength, tlength], e.g. num = 100, xlength = 50, tlength = 11
  forward and backward: write the same data, with different file names and equation names
  will choose time instances during dataloading.
  '''
  for problem_type in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, problem_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, xs, rhos in zip(all_params, all_xs, all_rhos):
        if np.sum(rhos) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, problem_type, params)
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, xs.shape, rhos.shape, xs.shape, rhos.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = xs, cond_v = rhos, qoi_k = xs, qoi_v = rhos)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")


def write_mfc_rhoparam_solve_tfrecord(name, eqn_type, all_params, all_xs, all_gs, all_rhos, problem_types):
  '''
  rhos: [num, xlength, tlength], e.g. num = 100, xlength = 50, tlength = 11
  '''
  for problem_type in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, problem_type)
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, xs, gs, rhos in zip(all_params, all_xs, all_gs, all_rhos):
        if np.sum(rhos) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, problem_type, params)
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, xs.shape, gs.shape, xs.shape, rhos.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = xs, cond_v = gs, qoi_k = xs, qoi_v = rhos)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")



def write_mfc_gparam_hj_tfrecord(name, eqn_type, all_params, all_txs, all_rhos, problem_types, nt, nx):
  '''
  txs: (num, nt * nx, 2)
  rhos: (num, nt * nx, 1)
  '''
  first_half_t = nt // 2
  
  if "forward11" in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward11")
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, txs, rhos in zip(all_params, all_txs, all_rhos):
        if np.sum(rhos) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, "forward11", params)
          cond_k = txs[:,:nx,:] # initial rho
          cond_v = rhos[:,:nx,:] # initial rho
          qoi_k = txs[:,-nx:,:] # terminal rho
          qoi_v = rhos[:,-nx:,:] # terminal rho
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if "forward12" in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward12")
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, txs, rhos in zip(all_params, all_txs, all_rhos):
        if np.sum(rhos) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, "forward12", params)
          cond_k = txs[:,:nx,:] # initial rho
          cond_v = rhos[:,:nx,:] # initial rho
          qoi_k = txs[:,first_half_t * nx:,:] # second half t rho
          qoi_v = rhos[:,first_half_t * nx:,:] # second half t rho
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if "forward22" in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward22")
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, txs, rhos in zip(all_params, all_txs, all_rhos):
        if np.sum(rhos) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, "forward22", params)
          cond_k = txs[:,:first_half_t * nx,:] # first half t rho
          cond_v = rhos[:,:first_half_t * nx,:] # first half t rho
          qoi_k = txs[:,first_half_t * nx:,:] # second half t rho
          qoi_v = rhos[:,first_half_t * nx:,:] # second half t rho
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")


def write_mfc_rhoparam_hj_tfrecord(name, eqn_type, all_params, 
                                   all_rhos_key, all_rhos_value, all_gs_key, all_gs_value, 
                                   problem_types, nt, nx):
  '''
  rhos_key: (num, nt * nx, 2)
  rhos_value: (num, nt * nx, 1)
  gs_key: (num, nx, 1)
  gs_value: (num, nx, 1)
  '''
  first_half_t = nt // 2

  if "forward11" in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward11")
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, rhos_key, rhos_value, gs_key, gs_value in zip(all_params, all_rhos_key, all_rhos_value, all_gs_key, all_gs_value):
        if np.sum(rhos_value) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, "forward11", params)
          cond_k = gs_key
          cond_v = gs_value
          qoi_k = rhos_key[:,-nx:,:] # terminal rho
          qoi_v = rhos_value[:,-nx:,:] # terminal rho
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")

  if "forward12" in problem_types:
    filename = "{}_{}_{}.tfrecord".format(name, eqn_type, "forward12")
    print(filename, flush=True)
    printed = False
    with tf.io.TFRecordWriter(filename) as writer:
      for params, rhos_key, rhos_value, gs_key, gs_value in zip(all_params, all_rhos_key, all_rhos_value, all_gs_key, all_gs_value):
        if np.sum(rhos_value) != np.nan:
          equation_name = "{}_{}_{}".format(eqn_type, "forward12", params)
          cond_k = gs_key
          cond_v = gs_value
          qoi_k = rhos_key[:,first_half_t * nx:,:] # second half t rho
          qoi_v = rhos_value[:,first_half_t * nx:,:] # second half t rho
          if not printed:
            print("example name: {}, cond_k.shape: {}, cond_v.shape: {}, qoi_k.shape: {}, qoi_v.shape: {}".format(
                      equation_name, cond_k.shape, cond_v.shape, qoi_k.shape, qoi_v.shape), flush=True)
            printed = True
          s_element= serialize_element(equation = equation_name, cond_k = cond_k, cond_v = cond_v, qoi_k = qoi_k, qoi_v = qoi_v)
          writer.write(s_element)
        else:
          raise Exception("NaN found!")