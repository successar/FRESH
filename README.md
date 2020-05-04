Learning to Faithfully Rationalize by Contruction
--------------------------------------------------

**Coming Soon**

Suggested use is to use Anaconda and do `pip install -r requirements.txt` .

1. `data` : Folder to store datasets
2. `Rationale_Analysis/models` : Folder to store allennlp models
    1. `classifiers` : Models that do actually learning 
    2. `saliency_scorer` : Takes a trained model and return saliency scorers for inputs
    3. `rationale_extractors` : Models that take saliency scores and generate rationales in form readable by `rationale_reader.py`
    4. `base_predictor.py` : Simple predictor to use with allennlp predict command as needed
3. `Rationale_Analysis/subcommands` : Subcommands to run saliency and rationale extractors since allennlp existingcommand semantics doesn't map quite as well to what we wanna do.
4. `Rationale_Analysis/training_config` : Contains jsonnet training configs to use with allennlp for each of the three types of models above.
5. `Rationale_Analysis/commands` : Actual bash scripts to run stuff.
6. `Rationale_Analysis/data/dataset_readers` : Contains dataset readers to work with Allennlp.
    1. `rationale_reader.py` : Code to load actual datasets (jsonl with 3 fields - document, query, label)
    2. `saliency_reader.py` : Read output of Saliency scorer to pass into rationale extractors.

Experimental Setup
------------------ 

1. Train any model 

```bash
CUDA_DEVICE=0 \
DATASET_NAME=<dataset name> \
CLASSIFIER=<classifier type> \
DATA_BASE_PATH=<path to data> \
EXP_NAME=<your experiment name> \
bash Rationale_Analysis/commands/model_train_script.sh
```

You `path to data` folder should contain three files - {train/dev/test}.jsonl . Each should be a list of dicts containing atleast two fields - 

```python
{
    "document" : str,
    "label" : str,
    "query" : Optional[str]
}
```
    
Output generated in `outputs/<classifier type>/<dataset name>/<your experiment name>/` .

2. Generate saliency scores

```bash
CUDA_DEVICE=0 \
DATASET_NAME=SST \
CLASSIFIER=<classifier type> \
DATA_BASE_PATH=Datasets/SST/data \
EXP_NAME=<your model-A exp name> \
SALIENCY=<saliency name> \
bash Rationale_Analysis/commands/saliency_script.sh
```
Output generate in `outputs/<classifier type>/SST/<your model-A exp name>/<saliency name>_saliency` .

3. Extract rationales from saliency

```bash
CUDA_DEVICE=0 \
DATASET_NAME=SST \
CLASSIFIER=<classifier type> \
EXP_NAME=<your model-A exp name> \
SALIENCY=<saliency name> \
RATIONALE=<rationale extraction type> \
RATIONALE_EXP_NAME=<your rationale exp name> \
bash Rationale_Analysis/commands/rationale_extractor_script.sh
```
Output generated in `outputs/<classifier type>/SST/<your model-A exp name>/<saliency name>_saliency/<rationale extraction type>_rationale/<your rationale exp name>`.

**Note:** A global rationale type would also require setting `BATCH_SIZE` to an arbitrarily large number.

4. Train model b from extracted rationales
```bash
CUDA_DEVICE=0 \
DATASET_NAME=SST \
CLASSIFIER=<classifier type> \
EXP_NAME=<your model-A exp name> \
SALIENCY=<saliency name> \
RATIONALE=<rationale extraction type> \
RATIONALE_EXP_NAME=<your exp name> \
bash Rationale_Analysis/commands/model_b.sh
```

5 (replacing 3+4). Extract Rationales and train model b.

```bash
CUDA_DEVICE=0 \
DATASET_NAME=SST \
CLASSIFIER=<classifier type> \
EXP_NAME=<your model-A exp name> \
SALIENCY=<saliency name> \
RATIONALE=<rationale extraction type> \
RATIONALE_EXP_NAME=<your exp name> \
bash Rationale_Analysis/commands/rationale_and_model_b_script.sh
```


Allowed values for now 

1. saliency name - file names in training_config/saliency_scorers
2. classifier type - "bert_classification"
3. rationale extraction type - filenames in training_config/rationale_extractors . Note each rationale extractor may also need some hyperparameter setting through env variables.
