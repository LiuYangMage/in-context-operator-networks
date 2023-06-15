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
import utils

runs = {"all data": {"stamp": "20230515-094404_1000000", "line": "solid", "color": "grey", "alpha": 1.0},
        "wrong operator": {"stamp": "fake_op", "line": "solid", "color": "black", "alpha": 1.0},
        "wrong demos": {"stamp": "20230522-234219_200000_fake_demo", "line": "dashed", "color": "red", "alpha": 1.0},
        "ODE 2": {"stamp":"20230522-234219_200000", "line": "dotted", "color": "r", "alpha": 1.0},
        "ODE 1,2": {"stamp": "20230524-165312_200000", "line": "dotted", "color": "orange", "alpha": 1.0},
        "ODE 2,3": {"stamp": "20230525-010119_200000", "line": "dashed", "color": "c", "alpha": 1.0}, 
        "ODE 1,2,3": {"stamp": "20230525-090937_200000", "line": "dashed", "color": "blue", "alpha": 1.0},
}


def myplot(analysis_folder, runs_plot, datadir, mode):

  plt.figure(figsize=(4,3))

  for label in runs_plot:
    stamp = runs[label]["stamp"]
    line = utils.linestyles[runs[label]["line"]]
    color = runs[label]["color"]
    alpha = runs[label]["alpha"]
    filename = f"{analysis_folder}/err_{stamp}_{datadir}_5_6.pickle"
    print(filename)
    with open(filename, 'rb') as file:
      results = pickle.load(file)
    ncoeffs = []
    err_mean = []
    for (str, ncoeff,), result in results.items():
      if mode in str:
        ncoeffs.append(ncoeff)
        err_mean.append(result["relative_error_mean"])
    x = [[i,j] for i,j in zip(ncoeffs, err_mean)]
    x.sort(key=lambda x: x[0])
    new_ncoeffs = np.array(x)[:,0]
    new_err_mean = np.array(x)[:,1]
    plt.plot(new_ncoeffs, new_err_mean, label=label, linestyle = line, color=color, alpha=alpha)

  plt.legend(ncols = 1)
  plt.xlabel('coefficient for new term ($b$)')
  plt.ylabel('average relative error')
  plt.xlim([FLAGS.xmin, FLAGS.xmax])
  yxrator = 0.55 if mode == "forward" else 1.5
  plt.ylim([0, FLAGS.xmax * yxrator])
  plt.title(f"{mode} problem", fontsize = 10)

  filename = f'{analysis_folder}/err_runs_on_{datadir}_{FLAGS.xmax}_{mode}.pdf'
  plt.savefig(filename, format = 'pdf', bbox_inches='tight')



def main(argv):

  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)

  if FLAGS.runs_plot is None:
    runs_plot = ['wrong operator', 'wrong demos', 'ODE 2', 'ODE 1,2', 'ODE 2,3', 'ODE 1,2,3', 'all data']

  myplot(FLAGS.analysis_folder, runs_plot, FLAGS.datadir, 'forward')
  myplot(FLAGS.analysis_folder, runs_plot, FLAGS.datadir, 'inverse')


if __name__ == '__main__':

  FLAGS = flags.FLAGS

  flags.DEFINE_string('analysis_folder', '../analysis/analysis0521-ood', 'the folder where analysis results are stored')
  flags.DEFINE_list('runs_plot', None, 'the runs to plot')
  flags.DEFINE_string('datadir', 'data0520_nt_odelinear3', 'the datadir to plot')
  flags.DEFINE_float('xmin', -0.3, 'the xmin for the plot')
  flags.DEFINE_float('xmax', 0.3, 'the xmax for the plot')
  

  app.run(main)
