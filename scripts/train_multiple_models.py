import subprocess
import os
import re

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    print(process)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

def train_across_domains(file_dir, config, model_type):
    d_list = ""
    for domain in os.listdir(file_dir):
        d_list += f"python -m spacy train {config} " \
                  f"--output ../trained_model/{model_type}/{domain} " \
                  f"--paths.train datasets/preprocessed/{domain}/results_only/train.spacy " \
                  f"--paths.dev datasets/preprocessed/{domain}/results_only/dev.spacy " \
                  f"-c ./scripts/custom_functions.py --gpu-id 0; "
    subprocess_cmd(d_list[-2])

def train_across_strats(file_dir, config, model_type):
    d_list = ""
    for strat in os.listdir(file_dir):
        print(strat)
        name = "strat_" + re.search("\d+\%",strat).group(0)
        print(name)
        os.system(f"python -m spacy train {config} " \
                  f"--output ../trained_models/{model_type}/all_domain_strats/{name} " \
                  f"--paths.train ../datasets/preprocessed/all_domains/training_stratifications/{strat} " \
                  f"--paths.dev ../datasets/preprocessed/all_domains/results_only/dev.spacy " \
                  f"-c ../scripts/custom_functions.py --gpu-id 0")
    #print(d_list[:-2])
    #subprocess_cmd(d_list[:-2])

if __name__ == "__main__":
    #train_across_domains("../datasets/preprocessed", "configs/ner_biobert.cfg", "ner")

    # ner
    #train_across_strats("../datasets/preprocessed/all_domains/training_stratifications", "../configs/ner_biobert.cfg", "ner")

    # rel
    train_across_strats("../datasets/preprocessed/all_domains/training_stratifications", "../configs/rel_biobert.cfg", "rel")


#def train_across_strats():

#def train_different_models

