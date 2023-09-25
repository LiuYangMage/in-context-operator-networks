# In-Context Operator Networks (ICON)

This folder contains the code associated with the paper titled ["In-Context Operator Learning with Data Prompts for Differential Equation Problems"](https://www.pnas.org/doi/10.1073/pnas.2310142120) published in *Proceedings of the National Academy of Sciences (PNAS)*.

## Environment Setup

### Docker
To facilitate the setup process, a Dockerfile is provided in this repository. Each user is required to build their own Docker image by following the instructions provided in [this guide](https://vsupalov.com/docker-shared-permissions/). Please replace `repo`, `tag`, and `xxx` with your own values.

To build the Docker image (you may want to replace `repo` and `tag`):

```
docker build - < Dockerfile -t repo:tag --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
```

To run the Docker container (Please navigate to your preferred directory first. Replace `repo` and `tag` with the ones you specified. You may also want to replace `xxx`):

```
docker run --gpus all --rm -itd --name xxx -v $(pwd):/workspace/ repo:tag bash
```

To attach to the Docker container (replace `xxx` with the one you specified):

```
docker attach xxx
```

To prevent disruptions due to timeouts, the following line is included in the Dockerfile:

```
RUN export PIP_DEFAULT_TIMEOUT=100
```

Please note that this Docker image does not have CUDA and cuDNN installed outside of pip. Therefore, TensorFlow will not be able to utilize the GPU. However, JAX and PyTorch will still be able to utilize the GPU. Since only tensorboard and tf.data.Dataset is used with TensorFlow, GPU usage for TensorFlow is not necessary.

### Pip

The pip install commands can be found in the Dockerfile.

## Data Preparation

The code for data preparation is located in the `data_generation/` folder. To generate the training data, in-distribution testing data, and out-of-distribution testing data, navigate to the `data_generation/` folder and execute the following commands: 

```
bash datagen.sh     # training data and in-distribution testing data
bash datagen_ood.sh # out-of-distribution testing data, including equations of new forms
```

## Training

The code for training is located in the current folder. 

To train with all training data, use the following command:

```
python3 run.py --problem hero --num_heads 8 --num_layers 6 --hidden_dim 256 --train_batch_size 32 --epochs 100 --train_data_dirs './data_generation/data0511a' --k_dim 3 --k_mode itx --tfboard --plot_num 16
```

In the paper, the effect of different training datasets was also studied. Execute `bash run_group.sh` to perform these experiments.

## Analysis

The code for analysis is located in the `analysis/` folder. To generate the results presented in the paper, navigate to the `analysis/` folder and execute the following commands:

```
bash analysis_ind.sh # in-distribution testing
bash analysis_len.sh # super-resolution and sub-resolution
bash analysis_ood.sh # out-of-distribution testing, including equations of new forms
```

To generate the figures presented in the paper, run the Python scripts and Jupyter notebooks with filenames starting with `plot_`. 

To run the analysis successfully, you may need to make modifications to the directory paths and checkpoint time stamps.

## Reference
[In-Context Operator Learning with Data Prompts for Differential Equation Problems](https://www.pnas.org/doi/10.1073/pnas.2310142120)
```
@article{yang2023context,
  title={In-context operator learning with data prompts for differential equation problems},
  author={Yang, Liu and Liu, Siting and Meng, Tingwei and Osher, Stanley J},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={39},
  pages={e2310142120},
  year={2023},
  publisher={National Acad Sciences}
}
```