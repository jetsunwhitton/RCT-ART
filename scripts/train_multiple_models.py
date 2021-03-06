import subprocess, os, re

def train_across_domains(file_dir, config, model_type, domain_cuts):
    """Trains models on different domain sets"""
    for domain in os.listdir(file_dir):
        print(domain)
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/biobert/{model_type}/{domain_cuts}/{domain} " \
                  f"--paths.train ../datasets/4_preprocessed/{domain_cuts}/{domain}/train.spacy " \
                  f"--paths.dev ../datasets/4_preprocessed/{domain_cuts}/{domain}/dev.spacy " \
                  f"-c ../scripts/custom_functions.py --gpu-id 0")


def train_across_strats(file_dir, config, model_type):
    """ trains different models on different all domains size stratifications"""
    for strat in os.listdir(file_dir):
        print(strat)
        name = "strat_" + re.search("\d+\%",strat).group(0)
        print(name)
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/biobert/{model_type}/all_domain_strats/{name} " \
                  f"--paths.train ../datasets/4_preprocessed/all_domains/stratifications/{strat} " \
                  f"--paths.dev ../datasets/4_preprocessed/all_domains/dev.spacy " \
                  f"-c ../scripts/custom_functions.py --gpu-id 0")


def train_across_models(configs):
    """Trains different BERT-based models from different configs"""
    for config in configs:
        model_base = os.path.basename(config).split(".")[0].split("_")[1]
        model_type = os.path.basename(config).split("_")[0]
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/{model_base}/{model_type}/all_domains " \
                  f"--paths.train ../datasets/4_preprocessed/all_domains/train.spacy " \
                  f"--paths.dev ../datasets/4_preprocessed/all_domains/dev.spacy " \
                  f"-c ../scripts/custom_functions.py --gpu-id 0")


if __name__ == "__main__":
    # train different language representations
    #model_configs = ["../configs/ner_biobert.cfg", "../configs/rel_biobert.cfg", "../configs/ner_scibert.cfg",
     #                "../configs/rel_scibert.cfg", "../configs/ner_roberta.cfg", "../configs/rel_roberta.cfg"]

    #train_across_models(model_configs)

    # train across strats

    # ner
    #train_across_strats("../datasets/4_preprocessed/all_domains/stratifications", "../configs/ner_biobert.cfg", "ner")

    # rel
    #train_across_strats("../datasets/4_preprocessed/all_domains/stratifications", "../configs/rel_biobert.cfg", "rel")

    # train across domains

    # ner
    #train_across_domains("../datasets/4_preprocessed/out_of_domain", "../configs/ner_biobert.cfg", "ner", "out_of_domain")

    # rel
    #train_across_domains("../datasets/4_preprocessed/out_of_domain", "../configs/rel_biobert.cfg", "rel", "out_of_domain")

    # ner
    #train_across_domains("../datasets/4_preprocessed/capped_for_comparison", "../configs/ner_biobert.cfg", "ner",
     #                    result_tab_0.csv"capped_for_comparison")

    # rel
    #train_across_domains("../datasets/4_preprocessed/capped_for_comparison", "../configs/rel_biobert.cfg", "rel",
     #                    "capped_for_comparison")

    # ner mixed domain comparison
   # train_across_domains("../datasets/4_preprocessed/capped_mix", "../configs/ner_biobert.cfg", "ner", "capped_mix")

    # rel mixed domain comparison
    #train_across_domains("../datasets/4_preprocessed/capped_mix", "../configs/rel_biobert.cfg", "rel", "capped_mix")




