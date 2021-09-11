import subprocess
import os
import re

def train_across_domains(file_dir, config, model_type):
    for domain in os.listdir(file_dir):
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/{model_type}/{domain} " \
                  f"--paths.train datasets/preprocessed/{domain}/results_only/train.spacy " \
                  f"--paths.dev datasets/preprocessed/{domain}/results_only/dev.spacy " \
                  f"-c ./scripts/custom_functions.py --gpu-id 0")


def train_across_strats(file_dir, config, model_type):
    for strat in os.listdir(file_dir):
        print(strat)
        name = "strat_" + re.search("\d+\%",strat).group(0)
        print(name)
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/biobert/{model_type}/all_domain_strats/{name} " \
                  f"--paths.train ../datasets/preprocessed/all_domains/training_stratifications/{strat} " \
                  f"--paths.dev ../datasets/preprocessed/all_domains/results_only/dev.spacy " \
                  f"-c ../scripts/custom_functions.py --gpu-id 0")


def train_across_models(configs):
    for config in configs:
        model_base = os.path.basename(config).split(".")[0]
        model_type = os.path.basename(config).split("_")[0]
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/{model_base}/{model_type}/all_domains " \
                  f"--paths.train ../datasets/preprocessed/all_domains/results_only/train.spacy " \
                  f"--paths.dev ../datasets/preprocessed/all_domains/results_only/dev.spacy " \
                  f"-c ../scripts/custom_functions.py --gpu-id 0")


if __name__ == "__main__":

    model_configs = ["../configs/ner_biobert.cfg", "../configs/rel_biobert.cfg", "../configs/ner_scibert.cfg",
                     "../configs/rel_scibert.cfg", "../configs/ner_roberta.cfg", "../configs/rel_roberta.cfg"]

    # train different language representations
    train_across_models(model_configs)

    #train_across_domains("../datasets/preprocessed", "configs/ner_biobert.cfg", "ner")

    # ner
    #train_across_strats("../datasets/preprocessed/all_domains/training_stratifications", "../configs/ner_biobert.cfg", "ner")

    # rel
    #train_across_strats("../datasets/preprocessed/all_domains/training_stratifications", "../configs/ner_biobert.cfg",
     #                   "ner")

    #train_across_strats("../datasets/preprocessed/all_domains/training_stratifications", "../configs/ner_biobert.cfg",
     #                   "rel")




#def train_across_strats():

#def train_different_models

