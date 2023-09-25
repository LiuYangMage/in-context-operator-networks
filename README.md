# In-Context Operator Networks (ICON)

This repository contains the code for the following papers:

## In-Context Operator Learning with Data Prompts for Differential Equation Problems 

This paper introduces “In-Context Operator Networks (ICON)” to learn operators from the prompted data and apply it to new questions, during the inference stage without any weight update. In the paper we trained a single neural network as a few-shot operator learner for a diversified type of problems, including forward and inverse problems of ODEs, PDEs, and mean-field control problems. We also show that it can generalize its learning capability to operators beyond the training distribution without fine-tuning. 

ICON draws inspiration from the success of large language models (LLM), e.g., GPT, for general natural language processing (NLP) tasks. We believe that it shows a promising direction to build large models for general scientific machine learning tasks.

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

