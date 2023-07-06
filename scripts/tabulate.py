"""
Tabulates RCT abstract result sentences. This module loads the NER and RE models,
which process input doc files, predictions are then used create enttiy tables
which are output a sCSV files.
"""
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
import operator, csv, os, io
from collections import defaultdict
# used by custom models
from evaluate import joint_ner_rel_evaluate as evaluate
# used by custom models
from rel_model import create_relation_model, create_classification_layer, create_instances
import pandas as pd


def named_entity_recognition(ner_model, input_docs):
    """Takes list of docs and extracts named entities in doc text and adds them to each doc"""
    print("|| Extracting entities")
    nlp = spacy.load(ner_model)  # the ner model is loaded here
    ent_processed_docs = []
    for doc in input_docs:
        input_text = doc.text  # doc.text extracts only text from doc file as input for model prediction
        doc.ents = nlp(input_text).ents  # model predicts entities and replaces the gold ents in doc
        ent_processed_docs.append(doc)
    return ent_processed_docs


def relation_extraction(rel_model, input_docs):
    """Takes list of docs and predicts entity pair relations, adding them to each doc"""
    print("|| Extracting relations")
    rel_nlp = spacy.load(rel_model)
    rel_processed_docs = []
    for doc in input_docs:
        doc._.rel = {}  # clear pre-exsiting rels in gold data
        for name, proc in rel_nlp.pipeline:  # take rel component from pipeline
            rel_preds = proc(doc)  # makes predicts probability of relation type between every entity pair
        rel_processed_docs.append(rel_preds)
    return rel_processed_docs


def tabulate_pico_entities(doc):
    """Tabulates the predicted result sentence entities using the extracted relations"""
    print("|| Tabulating docs")

    # create dictionaries for sorting entities into
    intv_dict = {"arm 1": set(), "arm 2": set()}
    meas_dict = defaultdict(lambda: defaultdict(set))

    # extract sorting infromation from relations into dictionaries
    for key in doc._.rel:
        rel = doc._.rel[key]  # get relation !!!!
        pred_rel = max(rel.items(), key=operator.itemgetter(1))  # selects relation type with highest probability
        if pred_rel[1] > 0.5:  # includes relation if above set threshold for probability
            entity = [ent.text for ent in doc.ents if ent.start == key[0]][0] # gets intv or outcome entity
            if pred_rel[0] == "A1_RES":
                intv_dict["arm 1"].add(entity)
                meas_dict[key[1]]["arm"].add(1)
            elif pred_rel[0] == "A2_RES":
                intv_dict["arm 2"].add(entity)
                meas_dict[key[1]]["arm"].add(2)
            else:
                meas_dict[key[1]]["outcomes"].add(entity)

    # create intv entity row of table
    arm_1 = ', '.join(sorted(str(x) for x in intv_dict["arm 1"])) # convert set to string
    arm_2 = ', '.join(sorted(str(x) for x in intv_dict["arm 2"])) # convert set to string
    intv_row = pd.DataFrame([["intervention", arm_1, arm_2]], columns=["outcome", "arm 1", "arm 2"])

    # seperate measures into a dictionary of respective outcomes and arms
    oc_dict = defaultdict(lambda: defaultdict(set))
    for k, v in meas_dict.copy().items():
        outcome = ', '.join(str(x) for x in v["outcomes"])
        meas = [ent.text for ent in doc.ents if ent.start == k][0]  # get full measure entity
        if v["arm"] == {1,2}:
            oc_dict[outcome]["arm_1"].add(meas)
            oc_dict[outcome]["arm_2"].add(meas)
            meas_dict.pop(k)
        elif v["arm"] == {1}:
            oc_dict[outcome]["arm_1"].add(meas)
            meas_dict.pop(k)
        elif v["arm"] == {2}:
            oc_dict[outcome]["arm_2"].add(meas)
            meas_dict.pop(k)

    for k, v in meas_dict.copy().items(): # sort measures with no associated intv in sentences with intvs
        outcome = ', '.join(str(x) for x in v["outcomes"])
        meas = [ent.text for ent in doc.ents if ent.start == k][0]  # get full measure entity
        if "arm_1" and "arm_2" in oc_dict.copy()[outcome]:
            oc_dict[outcome + ", total study group"]["arm_1"].add(meas)
            oc_dict[outcome + ", total study group"]["arm_2"].add(meas)
            meas_dict.pop(k)
        elif "arm_1" in oc_dict.copy()[outcome]:  # add measures to opposite arm if intv name missing
            oc_dict[outcome]["arm_2"].add(meas)
            meas_dict.pop(k)
        elif "arm_2" in oc_dict.copy()[outcome]:
            oc_dict[outcome]["arm_1"].add(meas)
            meas_dict.pop(k)

    for k, v in sorted(meas_dict.copy().items()): # sort measures in sentences with no intv by first mention
        outcome = ', '.join(str(x) for x in v["outcomes"])
        meas = [ent.text for ent in doc.ents if ent.start == k][0]  # get full measure entity
        if len(intv_dict) == 2:
            oc_dict[outcome + ", total study group"]["arm_1"].add(meas)
            oc_dict[outcome + ", total study group"]["arm_2"].add(meas)
            meas_dict.pop(k)
        elif outcome in oc_dict.copy():
            oc_dict[outcome]["arm_2"].add(meas)
            meas_dict.pop(k)
        else:
            oc_dict[outcome]["arm_1"].add(meas)
            meas_dict.pop(k)

    frames = [intv_row]
    for oc in oc_dict:
        if "arm_1" not in oc_dict[oc]:
            oc_dict[oc]["arm_1"].add("NR") # if meas missing, then included as not reported (NR)
        if "arm_2" not in oc_dict[oc]:
            oc_dict[oc]["arm_2"].add("NR")
        m_arm_1 = ', '.join(sorted(str(x) for x in oc_dict[oc]["arm_1"]))  # convert set to string
        m_arm_2 = ', '.join(sorted(str(x) for x in oc_dict[oc]["arm_2"]))  # convert set to string
        oc_row = pd.DataFrame([[oc, m_arm_1, m_arm_2]], columns=["outcome", "arm 1", "arm 2"])
        frames.append(oc_row)
    table = pd.concat(frames)
    return table

def output_csvs(dataframes, output_path):
    """
    Outputs list of dataframes as csv files
    """
    num = 0
    for df in dataframes:
        with io.open(f"{output_path}/result_tab_{num}.csv", 'w', encoding='utf-8') as output:
            df.to_csv(output)
            num += 1

if __name__ == "__main__":
    # tabulate predictions from different models
    doc_path = "../datasets/4_preprocessed/all_domains/test.spacy"
    seed_config_path = "../configs/ner/biobert"
    model_bases = ["biobert", "scibert", "roberta"]

   # for model_base in model_bases:
        #print(model_base)
       # for seed_run in os.listdir(seed_config_path):
           # print(seed_run)
          #  nlp = spacy.blank("en")
          #  doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
          #  docs = doc_bin.get_docs(nlp.vocab)
           # ner_preds = named_entity_recognition(f"../trained_models/{model_base}/ner/all_domains/{seed_run}/model-best", docs)
          #  rel_preds = relation_extraction(f"../trained_models/{model_base}/rel/all_domains/{seed_run}/model-best", ner_preds)
          #  dfs = []
          #  for doc in rel_preds:
           #     dfs.append(tabulate_pico_entities(doc))
          #  out_path = f"../output_tables/all_domains_{model_base}/{seed_run}"
          #  os.mkdir(out_path)
          #  output_csvs(dfs, out_path)

    # tabulate gold tables from test set
    #nlp = spacy.load("../trained_models/biobert/rel/all_domains/model-best")
    #doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
    #docs = doc_bin.get_docs(nlp.vocab)
    #dfs = []
    #count = 0
    #for doc in docs:
     #   print(count)
      #  dfs.append(tabulate_pico_entities(doc))
       # count += 1
    #output_csvs(dfs, "../datasets/5_gold_tables/all_domains")

    # tabulate predictions from different training size strats

    #for strat in os.listdir("../trained_models/biobert/ner/all_domain_strats"):
     #   print(strat)
      #  for seed_run in os.listdir(seed_config_path):
       #     nlp = spacy.blank("en")
        #    doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
         #   docs = doc_bin.get_docs(nlp.vocab)
          #  ner_preds = named_entity_recognition(f"../trained_models/biobert/ner/all_domain_strats/{strat}/{seed_run}/model-best", docs)
           # rel_preds = relation_extraction(f"../trained_models/biobert/rel/all_domain_strats/{strat}/{seed_run}/model-best", ner_preds)
           # dfs = []
            #for doc in rel_preds:
           #     dfs.append(tabulate_pico_entities(doc))
            #try:
             #   os.mkdir(f"../output_tables/all_domains_strats/train_{strat}")
            #except:
             #   pass
            #out_path = f"../output_tables/all_domains_strats/train_{strat}/{seed_run}"
            #try:
             #   os.mkdir(out_path)
            #except:
             #   pass
            #output_csvs(dfs, out_path)

# tabulate predictions from different domains
    domaincuts = ["out_of_domain","capped_for_comparison","capped_mix"]
    for domaincut in domaincuts:
        print(domaincut)
        doc_path = os.listdir(f"../datasets/4_preprocessed/{domaincut}")
        for domain in os.listdir(f"../datasets/4_preprocessed/{domaincut}"):
            print(domain)
            for seed_run in os.listdir(seed_config_path):
                print(seed_run)
                nlp = spacy.blank("en")
                doc_bin = DocBin(store_user_data=True).from_disk(f"../datasets/4_preprocessed/{domaincut}/{domain}/test.spacy")
                docs = doc_bin.get_docs(nlp.vocab)
                ner_preds = named_entity_recognition(f"../trained_models/biobert/ner/{domaincut}/{domain}/{seed_run}/model-best", docs)
                rel_preds = relation_extraction(f"../trained_models/biobert/rel/{domaincut}/{domain}/{seed_run}/model-best", ner_preds)
                dfs = []
                for doc in rel_preds:
                    dfs.append(tabulate_pico_entities(doc))
                try:
                    os.mkdir(f"../output_tables/{domaincut}/{domain}")
                except:
                    pass
                out_path = f"../output_tables/{domaincut}/{domain}/{seed_run}"
                try:
                    os.mkdir(out_path)
                except:
                    pass
                output_csvs(dfs, out_path)# for predicted
            #output_csvs(dfs, f"../datasets/5_gold_tables/out_of_domain/{domain}") # for gold tables

    # create capped_for_comparison preds or gold (ignore models and use tabulate function straight on docs for gold)
    #for domain in os.listdir("../datasets/4_preprocessed/capped_for_comparison"):
       #print(domain)
       #doc_path = f"../datasets/4_preprocessed/capped_for_comparison/{domain}/test.spacy"
       #nlp = spacy.blank("en")
       #doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
       #docs = doc_bin.get_docs(nlp.vocab)
      #ner_preds = named_entity_recognition(f"../trained_models/biobert/ner/capped_for_comparison/{domain}/model-best",
        #                                docs)
      #rel_preds = relation_extraction(f"../trained_models/biobert/rel/capped_for_comparison/{domain}/model-best",
        #                           ner_preds)
       #dfs = []
       #for doc in docs:
        #   dfs.append(tabulate_pico_entities(doc))
       #output_csvs(dfs, f"../datasets/5_gold_tables/capped_for_comparison/{domain}") # for gold tables
     # output_csvs(dfs, f"../output_tables/capped_for_comparison/{domain}") # for predicted


    # build incremental domain sets
    #for domain in os.listdir("../datasets/4_preprocessed/capped_mix"):
        #print(domain)
        #doc_path = f"../datasets/4_preprocessed/capped_mix/{domain}/test.spacy"
        #nlp = spacy.blank("en")
        #doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
        #docs = doc_bin.get_docs(nlp.vocab)
        #ner_preds = named_entity_recognition(f"../trained_models/biobert/ner/capped_mix/{domain}/model-best",
         #                                    docs)
        #rel_preds = relation_extraction(f"../trained_models/biobert/rel/capped_mix/{domain}/model-best",
         #                               ner_preds)
        #dfs = []
        #for doc in docs:
         #   dfs.append(tabulate_pico_entities(doc))
        #output_csvs(dfs, f"../datasets/5_gold_tables/capped_mix/{domain}")  # for gold tables
      # output_csvs(dfs, f"../output_tables/capped_mix/{domain}") # for predicted



