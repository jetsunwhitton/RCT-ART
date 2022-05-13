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
from pandas import DataFrame


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

    #iterate through docs and extract relations
    rel_dict = {"arm_1":[], "arm_2":[], "outcomes":[]}
    for key in doc._.rel:
        rel = doc._.rel[key] # get relation !!!!
        pred_rel = max(rel.items(),key=operator.itemgetter(1))  # selects relation type with highest probability

        if pred_rel[1] > 0.5:  # includes relation if above set threshold for probability
            if pred_rel[0] == "A1_RES": rel_dict["arm_1"].append((pred_rel,key))
            elif pred_rel[0] == "A2_RES": rel_dict["arm_2"].append((pred_rel,key))
            else: rel_dict["outcomes"].append((pred_rel,key))
    # structure entities as dictionary using relations
    final_dict = defaultdict(dict)
    if rel_dict["outcomes"] != []:
        for oc_rel,ockey in rel_dict["outcomes"]:
            oc_description = list(filter(lambda x: x.start == ockey[0], doc.ents))[0].text  # gets arm 1 entity
            for a1_rel,a1key in rel_dict["arm_1"]:
                if ockey[1] == a1key[1]:
                    final_dict["intervention"]["Arm 1"] = list(filter(lambda x: x.start == a1key[0], doc.ents))[0].text  # gets arm 1 entity
                    final_dict[oc_description]["Arm 1"] = list(filter(lambda x: x.start == a1key[1], doc.ents))[0].text  # gets result entity
            for a2_rel, a2key in rel_dict["arm_2"]:
                if ockey[1] == a2key[1]:
                    final_dict["intervention"]["Arm 2"] = list(filter(lambda x: x.start == a2key[0], doc.ents))[0].text  # gets arm 2 entity
                    final_dict[oc_description]["Arm 2"] = list(filter(lambda x: x.start == a2key[1], doc.ents))[0].text  # gets result entity
    else:
        for a1_rel, a1key in rel_dict["arm_1"]:
            final_dict["OC unspecified"]["Arm 1"] = list(filter(lambda x: x.start == a1key[1], doc.ents))[0].text  # gets result entity
            final_dict["intervention"]["Arm 1"] = list(filter(lambda x: x.start == a1key[0], doc.ents))[0].text  # gets arm 1 entity
        for a2_rel, a2key in rel_dict["arm_2"]:
            final_dict["OC unspecified"]["Arm 2"] = list(filter(lambda x: x.start == a2key[1], doc.ents))[0].text  # gets result entity
            final_dict["intervention"]["Arm 2"] = list(filter(lambda x: x.start == a2key[0], doc.ents))[0].text  # gets arm 2 entity

    # create dataframe from dictionary
    df = DataFrame.from_dict(final_dict, orient='columns', dtype=None, columns=None).transpose()
    columns_titles = ["Arm 1","Arm 2"]
    df = df.reindex(columns=columns_titles)
    df.index.name = 'Outcomes'
    df.reset_index(inplace=True)
    df.drop_duplicates(subset=["Arm 1", "Arm 2"], keep='first', inplace=True) #  don't allow duplicates of same result tuple
    return df


def output_csvs(dataframes, output_path):
    """
    Outputs list of dataframes as csv files
    """
    num = 0
    for df in dataframes:
        with io.open(f"{output_path}/result_tab_{num}.csv", 'w') as output:
            try:
                df.to_csv(output)
            except:
                print("CSV incompatible: ", df)
            num += 1

if __name__ == "__main__":
    # tabulate predictions from different models
    doc_path = "../datasets/preprocessed/all_domains/results_only/test.spacy"
    model_bases = ["biobert", "scibert", "roberta"]
    for model_base in model_bases:
        print(model_base)
        nlp = spacy.blank("en")
        doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
        docs = doc_bin.get_docs(nlp.vocab)
        ner_preds = named_entity_recognition(f"../trained_models/{model_base}/ner/all_domains/model-best", docs)
        rel_preds = relation_extraction(f"../trained_models/{model_base}/rel/all_domains/model-best", ner_preds)
        dataframes = tabulate_pico_entities(rel_preds)
        output_csvs(dataframes, f"../output_tables/all_domains_{model_base}")

