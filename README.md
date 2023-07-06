<p align="center"><img width="50%" src="logo.jpg" /></p>

## Randomised controlled trial abstract result tabulator


RCT-ART is an NLP pipeline built with spaCy for converting clinical trial result sentences into tables through jointly extracting 
intervention, outcome and outcome measure entities and their relations. The system is currently constrained to result sentences with
specific measures of an outcome for a specific intervention and does not extract comparative relationship (e.g. a relative decrease between
the study intervention and placebo). 

This repository contains custom pipes and models developed, trained and run using the spaCy library. These are defined and initiated 
through configs and custom scripts. 

In addition, we include all stages of our datasets from their raw format, gold-standard annotations, pre-processed spacy docs and output tables of the system, as well as the evaluation results of the system for its different NLP tasks across each pre-trained model.

### Running the system from Python ###
After cloning this repository and pip installing its dependencies from `requirements.txt`, the system can be run in two steps:

#### 1. Download and extract the trained models #### 

In the primary study of RCT-ART, we explored a number of BERT-based models in the development of the system. Here, we make available
the BioBERT-based named entity recognition (NER) and relation extraction (RE) models:

**Download models from [here](https://drive.google.com/file/d/1QzF6RC-x-keiLwcvTUZUT7YdOAz6WQtP/view?usp=sharing)**.

The `train_models` folder of the compression file should be extracted into the root of the cloned directory for the system scripts
to be able to access the models. 

#### 2a. Demo the system NLP tasks #### 
Once the model folder has been extracted, a streamlit demo of the system NER, RE and tabulation tasks can be run locally on your browser with the following command:
```bash
streamlit run scripts/demo.py
```
#### 2b. Process multiple RCT result sentences #### 
Alternatively, multiple result sentences can be processed by the system using `tabulate.py` in the scripts directory. 
Input sentences should be in the Doc format, with the sentences from the study available within `datasets/preprocessed`.


### Training new models for the system ###
The NER and RE models employed by RCT-ART were both trained using spaCy config files, where we defined their architectures and training hyper-parameters.
These are included in the `config` directory, with a config for each model type and the different BERT-based language representation models we explored in the development of the 
system. The simplest way to initiate spaCy model training is with the library's inbuilt commands (https://spacy.io/usage/training), passing in the paths of the config file, training set and development set. Below
are the commands we used to train the models made available with this repository:

#### spaCy cmd for training BioBERT-based NER model on all-domains dataset ####

```bash
python -m spacy train configs/seed_3.cfg --output ../trained_models/biobert/ner/all_domains --paths.train ../datasets/preprocessed/all_domains/results_only/train.spacy --paths.dev ../datasets/preprocessed/all_domains/results_only/dev.spacy -c ../scripts/custom_functions.py --gpu-id 0
```

#### spaCy cmd for training BioBERT-based RE model on all-domains dataset ####

```bash
python -m spacy train configs/rel_biobert.cfg --output ../trained_models/biobert/rel/all_domains  --paths.train ../datasets/preprocessed/all_domains/results_only/train.spacy --paths.dev ../datasets/preprocessed/all_domains/results_only/dev.spacy -c ../scripts/custom_functions.py --gpu-id 0
```

### Repository contents breakdown ###
The following is a brief description of the assets available in this repository.

#### configs ####
Includes the spaCy config files for training NER and RE models of the RCT-ART system. These files define the model architectures, including the BERT-base language representations. 
Three of BERT language representations were experimented with for each model in the main study of this sytem: BioBERT, SciBERT and RoBERTa. 

#### datasets ####
Includes all stages of the data used to train and test the RCT-ART models from raw to split gold-standard files in spaCy doc format. 

Before filtering and result sentence extraction, abstracts were sourced from the [EBM-NLP corpus](https://ebm-nlp.herokuapp.com/) and the annotated corpus from the [Trenta et al. study](http://arxiv.org/abs/1509.05209),
which explored automated information extraction from RCTs, and was a key reference for our study.

#### evaluation_results ####
Output txt files from the `evaluate.py` script, giving precision, recall and F1 scores for each of the system tasks across the various dataset cuts.

#### output_tables ####
Output csv files from the `tabulate.py` script, includes the predicted tables output by our system for each test result sentence. 

#### scripts ####
Below is a contents list of the repository scripts with brief descriptions. Full descriptions can be found at the 
head of each script. 

`custom_functions.py` -- helper functions for supporting key modules of system.

`data_collection.py` -- classes and functions for filtering the EBM-NLP corpus and result sentence preprocessing.

`demo.py` -- a browser-based demo of the RCT-ART system developed with spaCy and Streamlit (see above).

`entity_ruler.py` -- a script for rules-based entity recognition. Unused in final system, but made available for further development.

`evaluate.py` -- a set of function for evaluating the system across the NLP tasks: NER, RE, joint NER + RE and tabulation.

`preprocessing.py` -- a set of function for further data preprocessing after data collection and splitting data into train, test and dev sets.

`rel_model.py` -- defines the relation extraction model.

`rel_pipe.py` -- integrates the relation extraction model as a spaCy pipeline component.

`tabulate.py` -- run the full system by loading the NER and RE models and running their outputs through a tabulation function. Can be used on batches of RCT sentences to output batches of CSV files. 

`train_multiple_models.py` -- iterates through spaCy train commands with different input parameters allowing batches of models to be trained. 

#### Common issues ####
The transformer models of this system need a GPU with suitable video RAM -- in the primary study, they were trained and run on a GeForce RTX 3080 10GB.

There can be issues with the transformer library dependencies -- CUDA and pytorch. If an issue occurs, ensure CUDA 11.1 is installed on your system, and try reinstalling PyTorch with the following command:
```bash
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```


#### References ####
1. The relation extraction component was adapted from the following [spaCy project tutorial](https://github.com/explosion/projects/tree/v3/tutorials/rel_component).

2. The EBM-NLP corpus is accessible from [here](https://ebm-nlp.herokuapp.com/) and its publication can be found [here](https://arxiv.org/abs/1806.04185).

3. The glaucoma corpus can be found in the [Trenta et al. study](http://arxiv.org/abs/1509.05209).