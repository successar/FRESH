Learning to Faithfully Rationalize by Construction
--------------------------------------------------

**Coming Soon**

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

### Training Fresh Model (supp and pred) using thresholded rationales only.


### Training Fresh Model (supp, ext and pred) using thresholded rationales and extractor model.

### Training Lei et al or Bastings et al baseline models
