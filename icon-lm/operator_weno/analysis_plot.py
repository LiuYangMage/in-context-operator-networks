import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from absl import flags, app

def get_error(file_name_list):
  error_mean_list = []
  error_std_list = []
  for file_name in file_name_list:
    file = np.load(file_name)
    if len(file["label"].shape) == len(file["out"].shape):
      label = file["label"]
      pred = file["out"]
    elif len(file["label"].shape) == 4 and len(file["out"].shape) == 3:
      assert file["label"].shape[1] == 1
      label = file["label"][:,0,:,:] # remove the dimension.
      pred = file["out"]
    else:
      raise ValueError("Unknown shape of label")
    # both [..., 100, 1]
    error = np.mean(np.abs(label - pred), axis = (-2,-1))
    error_mean_list.append(np.mean(error))
    error_std_list.append(np.std(error))
  return error_mean_list, error_std_list

def get_error_tune(file_name):

  file = np.load(file_name)
  error_mean_list = []
  error_std_list = []
  label = file["label"][:,0,:,:] # remove the dimension. [bs, 100,1]
  pred_tune = file["pred_tune"][:,:,0,:,:] # [bs, step, 100, 1]
  for i in range(pred_tune.shape[1]):
    pred = pred_tune[:,i,:,:]
    error = np.mean(np.abs(label - pred), axis = (-2,-1))
    error_mean_list.append(np.mean(error))
    error_std_list.append(np.std(error))
  return error_mean_list, error_std_list

def plot_error_vs_num():
  model_name_dict = {'fno': "FNO", 'deepo': "DeepONet"}
  model_name = model_name_dict[FLAGS.model]

  plt.figure(figsize=(5,4))
    
  for coeff, color in zip(["0.20"], ['black']):
    error_mean, _ = get_error(["notune_{}_data_fix_{}_{}_{}.pkl.npz".format(FLAGS.model, coeff, coeff, coeff)])
    plt.semilogx([2], [error_mean[0]], "s",  color=color, subs = [], label = "$f = {}u^3+{}u^2+{}u$".format(coeff, coeff, coeff))

  errors_mean, errors_std = {}, {}
  num_list = [5, 10, 30, 100, 300, 1000]
  coeff_list = ['0.21', '0.25', '0.30']
  for coeff in coeff_list:
    for model in [FLAGS.model]:
        errors_mean[(model,coeff)], errors_std[(model,coeff)] = \
          get_error(
            [f'tune_{model}_data_fix_{coeff}_{coeff}_{coeff}.pkl_demonum_{num}.npz' for num in num_list]
            )

  for key, color in zip (coeff_list, ['red', 'blue', 'green']):
    plt.semilogx(num_list, errors_mean[(FLAGS.model, key)], "s-", 
            label="$f = {}u^3+{}u^2+{}u$".format(key, key, key), color=color, subs = [])
    
  for coeff, color in zip(["0.21", "0.25", "0.30"], ['red', 'blue', 'green']):
    error_mean, _ = get_error(["notune_{}_data_fix_{}_{}_{}.pkl.npz".format(FLAGS.model, coeff, coeff, coeff)])
    plt.semilogx([2,5], [error_mean[0], errors_mean[(FLAGS.model, coeff)][0]], "s--",  color=color, subs = [])

  for coeff in ["0.20", "0.21", "0.25", "0.30"]:
    error_mean, _ = get_error(["icon_data_fix_{}_{}_{}.pkl_demonum_5.npz".format(coeff, coeff, coeff)])
    label = "ICON with 5 examples" if coeff == "0.20" else None
    plt.axhline(y=error_mean, color='gray', linestyle='--', label = label)


  plt.xticks([2]+num_list, ["pretrained\n model"]+num_list)
  # plt.xlim(1,1100)
  plt.xlabel('# examples')
  plt.ylabel('average error')
  plt.title(model_name)
  plt.legend()
  plt.tight_layout()
  plt.savefig(f'error_{model_name}.pdf')

def plot_error_vs_steps():

  plt.figure(figsize=(5,4))
  step_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] + [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  for demonum in [5, 10, 30, 100, 300]:
    file_name = "tune_all_fno_data_fix_0.30_0.30_0.30.pkl_demonum_{}.npz".format(demonum)
    error_mean_list, error_std_list = get_error_tune(file_name)
    plt.semilogx(step_list[1:], error_mean_list[1:], 'o--', label = "{} examples".format(demonum))
  plt.legend()
  plt.xlabel('steps of fine-tuning')
  plt.ylabel('average error')
  plt.title("FNO, $f = 0.30u^3+0.30u^2+0.30u$")
  plt.tight_layout()
  plt.savefig("error_vs_steps.pdf")

def main(argv):
  plot_error_vs_num()
  plot_error_vs_steps()

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  flags.DEFINE_string('model', 'fno', 'Model to use')
  app.run(main)
