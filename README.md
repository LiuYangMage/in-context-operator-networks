# In-Context Operator Networks (ICON)

This repository contains the code for the following papers. Feel free to star this repository and cite our papers if you find it useful for your research. You can find the citation details below.

## In-Context Operator Learning with Data Prompts for Differential Equation Problems 

This paper aims to answer the following question: Can we build a large model for broad spectrum of scientific machine learning tasks, just as large language models (LLM) like GPT for natural language processing (NLP)?

Many scientific learning tasks can be seen as operator learning problems, where an operator transforms a "condition" function into a "quantity of interest (QoI)" function. Past attempts have been made to approximate operators with neural networks. However, in these methods, one neural network is limited to approximating one operator, requiring retraining, at least fine-tuning, for even small changes in equations.

In this paper, we made a huge step forward by introducing "In-Context Operator Networks (ICON)". A distinguishing feature of ICON is its ability to learn operators from data prompts, during the inference phase without weight adjustments. In essence, this transformer-based model is trained to act as a generalist operator learner rather than being tuned as a specific operator approximator, so that a single model can tackle a variety of tasks involving diverse operators. 

As an example, in this paper, a single neural network adeptly manages 19 distinct problem types, encompassing forward and inverse ODE, PDE, and mean field control problems, each spanning a continuous spectrum of operators. The learning capability is even generalized to operators beyond the training distribution, without any fine-tuning. 


ICON draws inspiration from the success of LLM for general NLP tasks. We believe that it shows a promising direction to build large models for general scientific machine learning tasks.

The paper is published in [*Proceedings of the National Academy of Sciences (PNAS)*](https://www.pnas.org/doi/10.1073/pnas.2310142120). Code are in folder `icon/`. See more details in `icon/README.md`.


## Fine-Tune Language Models as Multi-Modal Differential Equation Solvers

We transform the in-context operator learning into a multi-modal framework by introducing "captions" as a means to incorporate human knowledge about the operator, in the form of natural language descriptions and equations.

Moreover, we introduce a novel approach, namely "ICON-LM", to train a language-model-like architecture, or directly fine-tune existing language models, for in-context operator learning. Inspired by "next token prediction" in language models, ICON-LM is trained via "next function prediction". ICON-LM significantly outperforms the vanilla ICON model, in that it achieves better accuracy with about half of the parameters, with less training time and similar memory consumption.

By bridging language models with operator learning and data-driven differential equation solvers, we have not only achieved substantial advancements in this specific domain, but also opened up a new avenue for the application of language models in scientific machine learning, a realm that remains largely under-explored.

See the paper for more details:
https://arxiv.org/pdf/2308.05061.pdf. Code are in folder `icon-lm/`. See more details in `icon-lm/README.md`.

## PDE Generalization of In-Context Operator Networks: A Study on 1D Scalar Nonlinear Conservation Laws

In this paper, we present a detailed methodology for solving PDE problems with ICON, and show how a single ICON model can make forward and reverse predictions for different equations with different strides, provided with appropriately designed data prompts. This is exemplified through a study on 1D scalar nonlinear conservation laws, a family of PDEs with temporal evolution. 

We show the positive evidence that ICON can generalize well to PDEs with new forms without any fine-tuning. In particular, an ICON model trained on conservation laws with cubic flux functions can generalize well to some other flux functions of more general forms, without fine-tuning.

See the paper for more details:
https://arxiv.org/pdf/2401.07364.pdf. Code are in folder `icon-lm/`. See more details in `icon-lm/README.md`.

## Reference:
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

@article{yang2023FineTune,
  title={Fine-Tune Language Models as Multi-Modal Differential Equation Solvers},
  author={Yang, Liu and Meng, Tingwei and Liu, Siting and Osher, Stanley J},
  journal={arXiv preprint arXiv:2308.05061},
  year={2023}
}

@article{yang2024pde,
  title={PDE Generalization of In-Context Operator Networks: A Study on 1D Scalar Nonlinear Conservation Laws},
  author={Yang, Liu and Osher, Stanley J},
  journal={arXiv preprint arXiv:2401.07364},
  year={2024}
}
```

