# In-Context Operator Networks (ICON)

This repository contains the code for the following papers. Feel free to star this repository and cite our papers if you find it useful for your research. You can find the citation details below.

## In-Context Operator Learning with Data Prompts for Differential Equation Problems 

This paper aims to answer the following question: Can we build a large model for broad spectrum of scientific machine learning tasks, just as large language models (LLM) like GPT for natural language processing (NLP)?

Many scientific learning tasks can be seen as operator learning problems, where an operator transforms a "condition" function into a "quantity of interest (QoI)" function. Past attempts have been made to approximate operators with neural networks. However, in these methods, one neural network is limited to approximating one operator, requiring retraining, at least fine-tuning, for even small changes in equations.

In this paper, we made a huge step forward by introducing "In-Context Operator Networks (ICON)". A distinguishing feature of ICON is its ability to learn operators from data prompts, during the inference phase without weight adjustments. In essence, this transformer-based model is trained to act as a generalist operator learner rather than being tuned as a specific operator approximator, so that a single model can tackle a variety of tasks involving diverse operators. 

As an example, in this paper, a single neural network adeptly manages 19 distinct problem types, encompassing forward and inverse ODE, PDE, and mean field control problems, each spanning a continuous spectrum of operators. The learning capability is even generalized to operators beyond the training distribution, without any fine-tuning. 


ICON draws inspiration from the success of LLM for general NLP tasks. We believe that it shows a promising direction to build large models for general scientific machine learning tasks.

The paper is published in [*Proceedings of the National Academy of Sciences (PNAS)*](https://www.pnas.org/doi/10.1073/pnas.2310142120). Code are in folder `icon/`. See more details in `icon/README.md`.



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

@article{yang2023prompting,
  title={Prompting In-Context Operator Learning with Sensor Data, Equations, and Natural Language},
  author={Yang, Liu and Meng, Tingwei and Liu, Siting and Osher, Stanley J},
  journal={arXiv preprint arXiv:2308.05061},
  year={2023}
}
```

