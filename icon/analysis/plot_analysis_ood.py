import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from absl import app, flags, logging
import sys
sys.path.append('../')
from jax.config import config
import tensorflow as tf
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
tf.config.set_visible_devices([], device_type='GPU')

from plot_analysis_utils import label_map


runs = {
        "hero_v4": {"stamp":"20230515-094404_1000000"},
}

testdatas = {
                "data0520_ood_odeconst_light": "ode_auto_const",
                "data0520_ood_odelinear1_light": "ode_auto_linear1",
                "data0520_ood_odelinear2_light": "ode_auto_linear2",
                "data0520_ood_pdeporous_randbdry_light": "pde_porous_spatial",
                "data0520_ood_seriesdamped_light": "series_damped_oscillator",

                "data0520_ood_odeconst": "ode_auto_const",
                "data0520_ood_odelinear1": "ode_auto_linear1",
                "data0520_ood_odelinear2": "ode_auto_linear2",
                "data0520_ood_pdeporous_randbdry": "pde_porous_spatial",
                "data0520_ood_seriesdamped": "series_damped_oscillator",
             }

in_dist_coeff1_range = {
   "ode_auto_const": (0.5, 1.5),
   "ode_auto_linear1": (0.5, 1.5),
   "ode_auto_linear2": (0.5, 1.5),
   "pde_porous_spatial": (0.5, 1.5),
   "series_damped_oscillator": (-1.0, 5.0),
}

in_dist_coeff2_range = {
   "ode_auto_const": (-1.0, 1.0),
   "ode_auto_linear1": (-1.0, 1.0),
   "ode_auto_linear2": (-1.0, 1.0),
   "pde_porous_spatial": (-2.0, 2.0),
   "series_damped_oscillator": (-1.0, 5.0),
}


nxs = {
   "ode_auto_const": 39,
   "ode_auto_linear1": 39,
   "ode_auto_linear2": 39,
   "pde_porous_spatial": 39,
   "series_damped_oscillator": 39,
}

nys = {
   "ode_auto_const": 30,
   "ode_auto_linear1": 30,
   "ode_auto_linear2": 30,
   "pde_porous_spatial": 30,
   "series_damped_oscillator": 30,
}

xylim = {
   "ode_auto_const": {'xmax': None},
   "ode_auto_linear1": {'xmax': None},
   "ode_auto_linear2": {'xmax': 3},
   "pde_porous_spatial": {'xmax': 4},
   "series_damped_oscillator": {'xmax': None},
}

xylabel = {
   "ode_auto_const": {'x': r'$a_1$', 'y': r'$a_2$'},
   "ode_auto_linear1": {'x': r'$a_1$', 'y': r'$a_2$'},
   "ode_auto_linear2": {'x': r'$a_1$', 'y': r'$a_2$'},
   "pde_porous_spatial": {'x': r'$a$', 'y': r'$c$'},
   "series_damped_oscillator": {'x': 'decay', 'y': 'relative error'},
}


def read_file(prob_type, analysis_folder, stamp, datadir, demo_num_begin, demo_num_end):
  with open(f"{analysis_folder}/err_{stamp}_{datadir}_{demo_num_begin}_{demo_num_end}.pickle", 'rb') as file:
    results = pickle.load(file)

  coeff1s_fwd, coeff1s_inv = [], []
  coeff2s_fwd, coeff2s_inv = [], []
  rel_err_means_fwd, rel_err_means_inv = [], []
  for key, value in results.items():
    if prob_type not in key[0]:
      print(key)
      print('error: not this type of problem')
    else:
      # coeffs = key.split("_")
      coeff1_buck = key[1]
      coeff2_buck = key[2]
      if "forward" in key[0]:  # forward problem
        coeff1s_fwd.append(coeff1_buck)
        coeff2s_fwd.append(coeff2_buck)
        rel_err_means_fwd.append(value['relative_error_mean'])
      elif "inverse" in key[0]:  # inverse problem
        coeff1s_inv.append(coeff1_buck)
        coeff2s_inv.append(coeff2_buck)
        rel_err_means_inv.append(value['relative_error_mean'])
      else:
        print('error: neither forward nor inverse')
  return coeff1s_fwd, coeff2s_fwd, rel_err_means_fwd, coeff1s_inv, coeff2s_inv, rel_err_means_inv



def plot_heatmap(prob_type, coeff1s, coeff2s, rel_err_means, filename, nx, ny, xlabel, ylabel, title, 
                 in_dist_coeff1_range, in_dist_coeff2_range):
    x = [[i,j,k] for i,j,k in zip(coeff1s, coeff2s, rel_err_means)]
    x.sort(key=lambda x: x[0:2])
    new_amins = np.array(x)[:,0].reshape(nx, ny)
    new_bmins = np.array(x)[:,1].reshape(nx, ny)
    new_err_mean = np.array(x)[:,2].reshape(nx, ny)
    fig, ax = plt.subplots(figsize=(4,3))
    ood_coeff1_min = np.min(new_amins)
    ood_coeff1_max = np.max(new_amins)
    ood_coeff2_min = np.min(new_bmins)
    ood_coeff2_max = np.max(new_bmins)
    ood_coeff1_gap = (ood_coeff1_max - ood_coeff1_min) / (nx-1)
    ood_coeff2_gap = (ood_coeff2_max - ood_coeff2_min) / (ny-1)
    print(f'title: {title}, {ood_coeff1_min}-{ood_coeff1_gap:.3f}-{ood_coeff1_max}, {ood_coeff2_min}-{ood_coeff2_gap:.3f}-{ood_coeff2_max}')
    im = ax.pcolormesh(new_amins + ood_coeff1_gap/2, new_bmins + ood_coeff2_gap/2, 
                       new_err_mean, cmap='bwr', vmin = 0, vmax = 0.1)
    in_dist_coeff1_min = in_dist_coeff1_range[0]
    in_dist_coeff1_len = in_dist_coeff1_range[1] - in_dist_coeff1_range[0]
    in_dist_coeff2_min = in_dist_coeff2_range[0]
    in_dist_coeff2_len = in_dist_coeff2_range[1] - in_dist_coeff2_range[0]
    rect = patches.Rectangle((in_dist_coeff1_min, in_dist_coeff2_min), 
                             in_dist_coeff1_len, in_dist_coeff2_len, 
                             linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin = ood_coeff1_min
    xmax= ood_coeff1_max + ood_coeff1_gap
    ymin = ood_coeff2_min
    ymax = ood_coeff2_max + ood_coeff2_gap
    if xylim[prob_type]['xmax'] is not None:
      xmax = xylim[prob_type]['xmax']
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize = 10)
    cbar = fig.colorbar(im)
    plt.tight_layout()
    # plt.savefig('{}/err_ood_inverse.pdf'.format(analysis_folder))
    plt.savefig(filename)


def plot_1d_errs(coeff1s, rel_err_means, filename, xlabel, ylabel, title):
    plt.close('all')
    plt.figure(figsize=(4,3))
    x = [[i,k] for i,k in zip(coeff1s, rel_err_means)]
    x.sort(key=lambda x: x[0:1])
    new_amins = np.array(x)[:,0]
    new_err_mean = np.array(x)[:,1]
    agap = new_amins[1] - new_amins[0]
    new_amiddles = new_amins + agap/2
    fig, ax = plt.subplots(figsize=(4,3))
    plt.plot(new_amiddles, new_err_mean)
    ood_coeff1_min = np.min(coeff1s)
    ood_coeff1_max = np.max(coeff1s)
    ax.set_xlim(ood_coeff1_min, ood_coeff1_max)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize = 10)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')



def myplot(analysis_folder, runs_plot, datadirs_plot):
  for run in runs_plot:
    for datadir in datadirs_plot:
      print("run: ", run, "datadir: ", datadir)
      demo_num_begin = 5
      demo_num_end = 6
      stamp = runs[run]["stamp"]
      prob_type = testdatas[datadir]
      title_fwd = label_map[prob_type + "_forward"]['legend']
      title_inv = label_map[prob_type + "_inverse"]['legend']
      title_fwd = f'{title_fwd}\nrelative error'
      title_inv = f'{title_inv}\nrelative error'
      filename_fwd = f'{analysis_folder}/err_run_{run}_on_{datadir}_forward.pdf'
      filename_inv = f'{analysis_folder}/err_run_{run}_on_{datadir}_inverse.pdf'
      nx = nxs[prob_type]
      ny = nys[prob_type]
      coeff1s_fwd, coeff2s_fwd, rel_err_means_fwd, coeff1s_inv, coeff2s_inv, rel_err_means_inv = read_file(
                                        prob_type, analysis_folder, stamp, datadir, demo_num_begin, demo_num_end)
      if "ood_seriesdamped" in filename_fwd:
        plot_1d_errs(coeff1s_fwd, rel_err_means_fwd, filename_fwd, xylabel[prob_type]['x'], xylabel[prob_type]['y'], title_fwd)
      else:
        plot_heatmap(prob_type, coeff1s_fwd, coeff2s_fwd, rel_err_means_fwd, filename_fwd, nx, ny, xylabel[prob_type]['x'], xylabel[prob_type]['y'], title_fwd, 
                     in_dist_coeff1_range[prob_type], in_dist_coeff2_range[prob_type])
      if "ood_seriesdamped" in filename_inv:
        plot_1d_errs(coeff1s_inv, rel_err_means_inv, filename_inv, xylabel[prob_type]['x'], xylabel[prob_type]['y'], title_inv)
      else:
        plot_heatmap(prob_type, coeff1s_inv, coeff2s_inv, rel_err_means_inv, filename_inv, nx, ny, xylabel[prob_type]['x'], xylabel[prob_type]['y'], title_inv, 
                     in_dist_coeff1_range[prob_type], in_dist_coeff2_range[prob_type])

  
def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  if FLAGS.runs_plot is None:
    runs_plot = [
                # "hero_v1",
                # "hero_v2",
                # "hero_v3",
                "hero_v4",
                # "ode-2",
                # "ode-1-2", 
                # "ode-2-3",
                # "ode-1-2-3", 
                # "odes-series", 
                # "odes-series-pdes", 
                # "odes-series-pdes-mfc", 
                ]
  else:
    runs_plot = FLAGS.runs_plot


  if FLAGS.datadirs_plot is None:
    if FLAGS.cost == "light":
      suffix = "_light"
    elif FLAGS.cost == "heavy":
      suffix = ""
    else:
      raise NotADirectoryError
    datadirs_plot = [
                      # "data0520_ood_odeconst" + suffix,
                      # "data0520_ood_odelinear1" + suffix,
                      "data0520_ood_odelinear2" + suffix,
                      "data0520_ood_pdeporous_randbdry" + suffix,
                      # "data0520_ood_seriesdamped" + suffix,
                      ]
  else:
    datadirs_plot = FLAGS.datadirs_plot


  myplot(FLAGS.analysis_folder, runs_plot, datadirs_plot)


if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_string('analysis_folder', '../analysis/analysis0521-ood', 'the folder where analysis results are stored')
  flags.DEFINE_string('cost', 'light', 'light or heavy datadirs')

  flags.DEFINE_list('runs_plot', None, 'the runs to plot')
  flags.DEFINE_list('datadirs_plot', None, 'the datadirs to plot')
  

  app.run(main)
