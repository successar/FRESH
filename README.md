Learning to Faithfully Rationalize by Construction
--------------------------------------------------

This repository contains for paper https://arxiv.org/abs/2005.00115 to appear in ACL2020.

### Installation

1. `conda install -n fresh python=3.8`
2. `conda activate fresh`
3. `pip install -r requirements.txt`
4. `python -m spacy download en`


### Structure of Repository


1. `Datasets` : Folder to store datasets. For each dataset, please run the processing code in Process.ipynb file in respective folders.

2. `Rationale_Analysis/models` : Folder to store allennlp models
    1. `classifiers` : Models that do actually learning 
    2. `saliency_scorer` : Takes a trained model and return saliency scorers for inputs
    3. `rationale_extractors` : Models that take saliency scores and generate rationales by thresholding.
    4. `rationale_generators` : Models that take in thresholded rationales and train an extractor model.
    4. `base_predictor.py` : Simple predictor to use with allennlp predict command as needed

3. `plugins` : Subcommands to run saliency and rationale extractors since allennlp existing command semantics doesn't map quite as well to what we wanna do.

4. `Rationale_Analysis/training_config` : Contains jsonnet training configs to use with allennlp for models described above.

5. `Rationale_Analysis/commands` : Actual bash scripts to run stuff.

6. `Rationale_Analysis/data/dataset_readers` : Contains dataset readers to work with Allennlp.
    1. `base_reader.py` : Code to load actual datasets (jsonl with 4 fields - document, query, label, Optional[rationale])
    2. `saliency_reader.py` : Read output of Saliency scorer to pass into rationale extractors.
    3. `extractor_reader.py` : Reader thresholded rationales to train extractor model.

### Some common variables

In the following run scripts, the environment variables below can take these values - 

- DATASET_NAME in {evinf, movies, SST, agnews, multirc}
- SALIENCY in {wrapper, simple_gradient} [Please note, wrapper is just another name for Attention based saliency]
- THRESHOLDER in {top_k, contiguous}
- MAX_LENGTH_RATIO in [0, 1] -- desired length of rationales
- BERT_TYPE in {bert-base-uncased, roberta-base, allenai/scibert_scivocab_uncased}

We use bert-base-uncased for {SST, agnews, movies}, roberta-base for multirc and scibert for evinf.

- HUMAN_PROB in [0, 1] -- amount of human supervision to use for rationales

## Method to run individual models

### Training Fresh Model (supp and pred) using thresholded rationales only.

```bash
CUDA_DEVICE=0 \
DATASET_NAME=$DATASET_NAME \
CLASSIFIER=bert_classification \
BERT_TYPE=$BERT_TYPE \
EXP_NAME=fresh \
MAX_LENGTH_RATIO=$MAX_LENGTH_RATIO \
SALIENCY=$SALIENCY \
THRESHOLDER=$THRESHOLDER \
EPOCHS=20 \
BSIZE=$BSIZE \
bash Rationale_Analysis/commands/fresh/fresh_script.sh
```

### Training Fresh Model (supp, ext and pred) using thresholded rationales and extractor model.

```bash
CUDA_DEVICE=0 \
DATASET_NAME=$DATASET_NAME \
CLASSIFIER=bert_classification \
BERT_TYPE=$BERT_TYPE \
EXP_NAME=fresh \
MAX_LENGTH_RATIO=$MAX_LENGTH_RATIO \
SALIENCY=$SALIENCY \
THRESHOLDER=$THRESHOLDER \
EPOCHS=20 \
BSIZE=$BSIZE \
HUMAN_PROB=$HUMAN_PROB \
bash Rationale_Analysis/commands/fresh/fresh_with_extractor_script.sh
```

### Training Lei et al model

MU/LAMBDA are hyperparameters for regularizer. Values we used after hyperparam search are in file Rationale_Analysis/default_values.json.

```console
CUDA_DEVICE=0 \
DATASET_NAME=$DATASET_NAME \
CLASSIFIER=bert_encoder_generator \
BERT_TYPE=$BERT_TYPE \
EXP_NAME=fresh \
MAX_LENGTH_RATIO=$MAX_LENGTH_RATIO \
EPOCHS=20 \
BSIZE=$BSIZE \
MU=$MU \
LAMBDA=$LAMBDA \
bash Rationale_Analysis/commands/encgen/experiment_script.sh
```

### Training Bastings et al model

```bash
CUDA_DEVICE=0 \
DATASET_NAME=$DATASET_NAME \
CLASSIFIER=bert_kuma_encoder_generator \
BERT_TYPE=$BERT_TYPE \
EXP_NAME=fresh \
MAX_LENGTH_RATIO=$MAX_LENGTH_RATIO \
EPOCHS=20 \
BSIZE=$BSIZE \
LAMBDA_INIT=1e-5 \
bash Rationale_Analysis/commands/encgen/experiment_script.sh
```

## Method to reproduce experiments in paper

### variation due to random seeds

1. For Lei et al,

```bash
CLASSIFIER=bert_encoder_generator \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type encgen/experiment_script.sh \
--all-data \
```

2. For Bastings et al,

```bash
CLASSIFIER=bert_kuma_encoder_generator \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type encgen/experiment_script.sh \
--all-data \
```

3. For Fresh,

```bash
CLASSIFIER=bert_classification \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type fresh/experiment_script.sh \
--all-data \
```

### variation due to rationale length

1. For Lei et al,

```bash
CLASSIFIER=bert_encoder_generator \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type encgen/experiment_script.sh \
--all-data \
--defaults-file Rationale_Analysis/second_cut_point.json
```

2. For Bastings et al,

```bash
CLASSIFIER=bert_kuma_encoder_generator \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type encgen/experiment_script.sh \
--all-data \
--defaults-file Rationale_Analysis/second_cut_point.json
```

3. For Fresh,

```bash
CLASSIFIER=bert_classification \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type fresh/experiment_script.sh \
--all-data \
--defaults-file Rationale_Analysis/second_cut_point.json
```

### variation due to human rationale supervision

1. For Lei et al Model,

```bash
for human_prob in 0.0 0.2 0.5 1.0;
do 
    DATASET_NAME=$DATASET_NAME \
    HUMAN_PROB=$human_prob \
    CLASSIFIER=bert_encoder_generator_human \
    python Rationale_Analysis/experiments/run_for_random_seeds.py \
    --script-type encgen/supervised_experiment_script.sh;
done;
```

2. For Fresh Model,

```bash
for human_prob in 0.0 0.2 0.5 1.0;
do 
    DATASET_NAME=$DATASET_NAME \
    HUMAN_PROB=$human_prob \
    CLASSIFIER=bert_classification \
    python Rationale_Analysis/experiments/run_for_random_seeds.py \
    --script-type fresh/fresh_with_extractor_script;
done;
```