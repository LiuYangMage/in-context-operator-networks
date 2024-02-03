# ICON-LM for Multi-Modal In-Context Operator Learning

This folder contains the code associated with the following two papers: 
- [Fine-Tune Language Models as Multi-Modal Differential Equation Solvers](https://arxiv.org/pdf/2308.05061.pdf). This paper focuses on improving the model architecture and training scheme for in-context operator learning.

- [PDE Generalization of In-Context Operator Networks: A Study on 1D Scalar Nonlinear Conservation Laws](https://arxiv.org/pdf/2401.07364.pdf). This paper focuses on solving PDE problems with in-context operator learning, espetially its generalization capability to new PDEs.

## Environment Setup

The YAML file `environment.yml` contains the environment setup for the code. To create the environment, run the following command:
```
conda env create -f environment.yml
```
which will create an environment named `icon`. You may need to update the `prefix` field in the YAML file to specify the location of the environment.

## Data Preparation

The captions are already stored in the `data_preparation/captions_1009` folder. The code for function data generation is located in the `data_preparation/` folder. Navigate to the `data_generation/` folder and run `bash datagen.sh`, which will generate the function data for the experiments. 

The generated data will be stored in the `data_preparation/data` folder. We moved data to `/home/shared/icon/data/data0910c` for our experiments.

## Training

All the in-context operator learning models shown in the paper are supported here, including (1) encoder-decoder ICON (baseline), (2) ICON-LM (ours), and (3) fine-tuning GPT-2. The run commands for training each model are listed in `run.sh`.

## Analysis and Visualization

The analysis code and run commands are located in the `analysis/` folder. The visualization code are located in the `plot_icon_lm/` folder.

## Classic Operator Learning Models

In the paper we compared ICON-LM with classic operator learning models, including FNO and DeepONet. The run commands for pretraining FNO and DeepONet are listed in `run.sh`. The code and commands for fine-tuning FNO and DeepONet are located in the `operator` folder.

