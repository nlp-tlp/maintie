# MaintIE Model Training

We've trained two distinct models on the annotated MaintIE corpora to facilitate automated information extraction:

- **Token-Classification**: [SpERT](https://github.com/lavis-nlp/spert)
- **Sequence-to-Sequence**: [REBEL](https://github.com/Babelscape/rebel)

This documentation describes our methodology and outlines steps for you to replicate our experiments, as discussed in the MaintIE paper.

## Fine-Tuning Approaches

We've assessed the performance of both model types using two distinct fine-tuning methodologies:

1. **Direct Fine-Tuning**: Training directly on the fine-grained corpus
2. **Sequential Fine-Tuning**: Initiating with pre-fine-tuning on the coarse-grained corpus, followed by fine-tuning on the fine-grained corpus.

The process is visually represented in the figure below:

![Annotation and Experiment Process](./annotation_and_experiment_process.png)

> Note: The primary objective is to evaluate the models using the fine-grained corpus. The coarse-grained corpus serves as a preliminary step, helping us gauge any enhancements in performance during subsequent fine-grained evaluation.

## Prerequisites

Each model comes with distinct dependencies. To ensure compatibility and smooth operation:

1. Ensure you're working with Python 3.8, as all our training was executed using this version.
2. Create a virtual environment for each model and install their specific dependencies (`requirements.txt`).
3. We used PyTorch with CUDA 12.1. This was installed via `pip install torch --index-url https://download.pytorch.org/whl/cu121` prior to installing the model specific dependencies.

### Model Repositories

For original versions and comprehensive details about the models, including training, evaluation, and execution processes, refer to their official repositories:

- **REBEL**: https://github.com/Babelscape/rebel
- **SpERT**: https://github.com/lavis-nlp/spert

### Data Preparation

To prepare the data for running the experiments, execute the `create_datasets.py` script. This will create 8 folders with datasets using different variants of the entity hierarchy (`s-0/1/2/3` and `g-0/1/2/3`).
The naming conventions of the files are `corpus-quality_entity-hierarchy-level`, e.g. `g_3` refers to the gold corpus (fine-grained expert-annotated) at level 3 of the entity hierarchy (224 classes). Similarly, `s_1` refers to the silver corpus at level 1 of the entity hierarchy (5 classes).

> Note: Both REBEL and SpERT use the exact same datasets, so performing this process once is sufficient for both models.

## [REBEL](https://github.com/Babelscape/rebel) Experiments

REBEL is a generative sequence-to-sequence model for relation extraction by end-to-end language generation. For more information please consult the models [repository](https://github.com/Babelscape/rebel).

> **Important: Ensure that you update the REBEL configurations to point to the correct absolute path on your machine.**

To reproduce our experiments with REBEL, first you need to create and activate a Python virtual environment and install the dependencies (`./rebel/requirements.txt`). To train the models, it is strongly suggested that you use a GPU.

For experiments fine-tuning the base REBEL model (e.g. `FG-0/1/2/3`), simply execute the commands shown in the table below after following the [instructions in the REBEL repository for downloading the REBEL base model](https://github.com/Babelscape/rebel#rebel-model-and-dataset) (`Rebel-large`).

For the coarse-grained and fine-grained fine-tuning process, you need to create a `maintie_model` which consists of fine-tuning the Rebel-large model on the silver corpus with 5 entity classes. This model is then saved and used as the foundational model for experiments (`CG+FG-0/1/2/3`). See the [next section](#Creating the REBEL MaintIE Base Model) for details.

#### Creating the REBEL MaintIE Base Model

To create the `maintie_model` for using the coarse-grained corpus to improve fine-grained information extraction, follow these steps (alternatively consult with the REBEL repository for further clarifications):

1. Train the MaintIE model by running `python train.py model=rebel_model data=maintie_s_1 train=maintie_s_1`. This will fine-tune the REBEL base model on the 5 class silver corpus.
2. Update the paths to the training outputs and hydra config in the `model_saving.py` script in `./src`. This will save the model for use in subsequent training processes.
3. Now use the `maintie_model` as a model parameter when running the CG+FG experiments.

#### Notes

- The original code was cloned from the REBEL repo with modifications made to work with the MaintIE scheme via configuration files and evaluation methods in `/src` scripts.

- You will need to change the base path of the configurations (`./rebel/conf`) to point to the local directory you are working in as they require absolute paths.

- ⚠️ If the error "Key 'config' is not in struct" occurs when testing the REBEL model, comment out the following line `checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)` (LN84) in the pytorch-lightning module `pytorch_lightning\core\saving.py` in your virtual environment.

#### Experiments

| Experiment | Training Command                                                         |
| ---------- | ------------------------------------------------------------------------ |
| FG-0       | `python train.py model=rebel_model data=maintie_g_0 train=maintie_g_0`   |
| FG-1       | `python train.py model=rebel_model data=maintie_g_1 train=maintie_g_1`   |
| FG-2       | `python train.py model=rebel_model data=maintie_g_2 train=maintie_g_2`   |
| FG-3       | `python train.py model=rebel_model data=maintie_g_3 train=maintie_g_3`   |
| CG+FG-0    | `python train.py model=maintie_model data=maintie_g_0 train=maintie_g_0` |
| CG+FG-1    | `python train.py model=maintie_model data=maintie_g_1 train=maintie_g_1` |
| CG+FG-2    | `python train.py model=maintie_model data=maintie_g_2 train=maintie_g_2` |
| CG+FG-3    | `python train.py model=maintie_model data=maintie_g_3 train=maintie_g_3` |

Note: `g` refers to gold corpus.

## [SpERT](https://github.com/lavis-nlp/spert) Experiments

SpERT is a span-based entity and relation transformer which jointly extracts entities and relations from text. It is a token-classification type model. For more information please consult the models [repository](https://github.com/lavis-nlp/spert).

The configuration files for reproducing our experiments with SpERT are located in `./spert/configs`. There are `train` and `eval` configuration files for each of the experiments. To facilitate the use of the coarse-grained corpus for pre-fine-tuning before fine-tuning on the fine-grained corpus, see the [section](#Creating the SpERT MaintIE Base Model) below. In general, the SpERT experiments are split into three stages:

1. Training and evaluating on the fine-grained corpus (gold; `maintie_g_*`),
2. Training on the coarse-grained corpus (silver; `maintie_s_*`) to produce MaintIE base-models for subsequent fine-tuning on the fine-grained corpus, and
3. Training and evaluating on the fine-grained corpus using the pre-fine-tuned MaintIE base model (`maintie_gs_*`). Note: `gs` refers to `gold silver` which denotes the combined use of the corpora.

#### Creating the SpERT MaintIE Base Model

To create the MaintIE base model simply run SpERT with the `maintie_s_*_train.conf` training configurations. This will produce a model pre-fine-tuned on the coarse-grained corpus. These pre-fine-tuned models are then used as the base model the `maintie_gs_*_train.conf` configurations. More details of the `s` configurations are given below.

For the untyped case, we use the base model trained without types. For the other three levels, we use the base model trained

- Level 0: base model is the SpERT model trained on the coarse-grained corpus with an output layer of 1 entity type. The dataset used is truncated to 1 entity type. `maintie_s_0_train.conf`
- Level 1: base model is the SpERT model trained on the coarse-grained corpus with an output layer of 5 entity types. `maintie_s_1_train.conf`
- Level 2: base model is the SpERT model trained on the coarse-grained corpus with an output layer of 32 entity types. `maintie_s_2_train.conf`
- Level 3: base model is the SpERT model trained on the coarse-grained corpus with an output layer of 224 entity types. `maintie_s_3_train.conf`

#### Notes

- After training each model, you will need to update the respective `eval.conf` `model_path` key to the path of the `final_model` for each experiment. This will use the trained model for evaluation.
- `gs` configs should have `s` models as `model_path` and `tokenizer_path`. The train/valid path should be to the `g` data. The `types` are to the `s` data incase any order has changed (the types link the types to the entities/relations; the order in train/valid don't matter as they are just strings.)

#### Experiments

| Experiment | Training Command                                             |
| ---------- | ------------------------------------------------------------ |
| FG-0       | `python ./spert.py train --config configs/maintie_g_0.conf`  |
| FG-1       | `python ./spert.py train --config configs/maintie_g_1.conf`  |
| FG-2       | `python ./spert.py train --config configs/maintie_g_2.conf`  |
| FG-3       | `python ./spert.py train --config configs/maintie_g_3.conf`  |
| CG+FG-0    | `python ./spert.py train --config configs/maintie_gs_0.conf` |
| CG+FG-1    | `python ./spert.py train --config configs/maintie_gs_1.conf` |
| CG+FG-2    | `python ./spert.py train --config configs/maintie_gs_2.conf` |
| CG+FG-3    | `python ./spert.py train --config configs/maintie_gs_3.conf` |

Note: `g` refers to gold corpus, `gs` refers to gold+silver corpus.
