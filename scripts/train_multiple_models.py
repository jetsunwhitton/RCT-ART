import subprocess, os, re

def train_across_domains(file_dir, model_type, domain_cuts):
    """Trains models on different domain sets"""
    for domain in os.listdir(file_dir):
        print(domain)
        for config_seed in os.listdir(f"../configs/ner/biobert"):
            os.system(f"python -m spacy train ../configs/{model_type}/biobert/{config_seed} " \
                      f"--output ../trained_models/biobert/{model_type}/{domain_cuts}/{domain}/{config_seed}  " \
                      f"--paths.train ../datasets/4_preprocessed/{domain_cuts}/{domain}/train.spacy " \
                      f"--paths.dev ../datasets/4_preprocessed/{domain_cuts}/{domain}/dev.spacy " \
                      f"-c ../scripts/custom_functions.py --gpu-id 0")


def train_across_strats(file_dir,model_type):
    """ trains different models on different all domains size stratifications"""
    for strat in os.listdir(file_dir):
        print(strat)
        name = "strat_" + re.search("\d+\%",strat).group(0)
        print(name)
        for config_seed in os.listdir(f"../configs/ner/biobert"):
            os.system(f"python -m spacy train ../configs/{model_type}/biobert/{config_seed} " \
                      f"--output ../trained_models/biobert/{model_type}/all_domain_strats/{name}/{config_seed} " \
                      f"--paths.train ../datasets/4_preprocessed/all_domains/stratifications/{strat} " \
                      f"--paths.dev ../datasets/4_preprocessed/all_domains/dev.spacy " \
                      f"-c ../scripts/custom_functions.py --gpu-id 0")


def train_across_models(model_config_paths):
    """Trains different BERT-based models from different configs"""
    for path in model_config_paths:
        model_type = os.path.basename(path).split("/")[-1]
        for model in os.listdir(path):
            for config_seed in os.listdir(f"{path}/{model}"):
                os.system(f"python -m spacy train {path}/{model}/{config_seed} " \
                          f"--output ../trained_models/{model}/{model_type}/all_domains/{config_seed} " \
                          f"--paths.train ../datasets/4_preprocessed/all_domains/train.spacy " \
                          f"--paths.dev ../datasets/4_preprocessed/all_domains/dev.spacy " \
                          f"-c ../scripts/custom_functions.py --gpu-id 0")


if __name__ == "__main__":
    # train different language representations

    #model_config_paths = ["../configs/ner", "../configs/rel",]
    #train_across_models(model_config_paths)

    # train across strats

    # ner
    #train_across_strats("../datasets/4_preprocessed/all_domains/stratifications", "ner")

    # rel
    #train_across_strats("../datasets/4_preprocessed/all_domains/stratifications", "rel")

    # train across domains

    # ner
    #train_across_domains("../datasets/4_preprocessed/out_of_domain", "ner", "out_of_domain")

    # rel
    #train_across_domains("../datasets/4_preprocessed/out_of_domain", "rel", "out_of_domain")

    # ner
    #train_across_domains("../datasets/4_preprocessed/capped_for_comparison", "ner", "capped_for_comparison")

    # rel
    #train_across_domains("../datasets/4_preprocessed/capped_for_comparison", "rel","capped_for_comparison")

    # ner mixed domain comparison
    #train_across_domains("../datasets/4_preprocessed/capped_mix", "ner", "capped_mix")

    # rel mixed domain comparison
    #train_across_domains("../datasets/4_preprocessed/capped_mix", "rel", "capped_mix")




