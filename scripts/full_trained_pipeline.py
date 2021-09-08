import spacy
from spacy.scorer import PRFScore
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
import operator
from collections import defaultdict, namedtuple, ChainMap
from scripts.evaluate import ner_rel_evaluate as evaluate # used by custom models
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances # used by custom models
import scripts.entity_ruler
from pandas import DataFrame
import csv
from spacy.scorer import PRFScore


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


def tabulate_pico_entities(nlp, input_docs):
    """Tabulates the predicted result sentence entities using the extracted relations"""
    print("|| Tabulating docs")
    pos_dataframes = []
    neg_dataframes = []

    #iterate through docs and extract relations
    for doc in input_docs:
        postives = {"arm_1":[], "arm_2":[], "outcomes":[]}
        negatives = {"arm_1":[], "arm_2":[], "outcomes":[]}
        for key in doc._.rel:
            rel = doc._.rel[key] # get relation
            pred_rel = max(rel.items(),key=operator.itemgetter(1))  # selects relation type with highest probability

            if pred_rel[1] > 0.8:  # includes relation if above set threshold for probability
                if pred_rel[0] == "A1_RES": postives["arm_1"].append((pred_rel,key))
                elif pred_rel[0] == "A2_RES": postives["arm_2"].append((pred_rel,key))
                else: postives["outcomes"].append((pred_rel,key))

            else:  # add relations for negative table
                if pred_rel[0] == "A1_RES": negatives["arm_1"].append((pred_rel,key))
                elif pred_rel[0] == "A2_RES": negatives["arm_2"].append((pred_rel,key))
                else: negatives["outcomes"].append((pred_rel,key))

        # structure entities as dictionary using relations
        pos_neg = 0
        for relations in [postives,negatives]:
            final_dict = defaultdict(dict)
            if relations["outcomes"] != []:
                for oc_rel,ockey in relations["outcomes"]:
                    oc_description = list(filter(lambda x: x.start == ockey[0], doc.ents))[0].text  # gets arm 1 entity
                    for a1_rel,a1key in relations["arm_1"]:
                        if ockey[1] == a1key[1]:
                            final_dict["intervention"]["Arm 1"] = list(filter(lambda x: x.start == a1key[0], doc.ents))[0].text  # gets arm 1 entity
                            final_dict[oc_description]["Arm 1"] = list(filter(lambda x: x.start == a1key[1], doc.ents))[0].text  # gets result entity
                    for a2_rel, a2key in relations["arm_2"]:
                        if ockey[1] == a2key[1]:
                            final_dict["intervention"]["Arm 2"] = list(filter(lambda x: x.start == a2key[0], doc.ents))[0].text  # gets arm 2 entity
                            final_dict[oc_description]["Arm 2"] = list(filter(lambda x: x.start == a2key[1], doc.ents))[0].text  # gets result entity
            else:
                for a1_rel, a1key in relations["arm_1"]:
                    if ockey[1] == a1key[1]:
                        final_dict["OC unspecified"]["Arm 1"] = list(filter(lambda x: x.start == a1key[1], doc.ents))[0].text  # gets result entity
                        final_dict["intervention"]["Arm 1"] = list(filter(lambda x: x.start == a1key[0], doc.ents))[0].text  # gets arm 1 entity
                for a2_rel, a2key in relations["arm_2"]:
                    if ockey[1] == a2key[1]:
                        final_dict["OC unspecified"]["Arm 2"] = list(filter(lambda x: x.start == a2key[1], doc.ents))[0].text  # gets result entity
                        final_dict["intervention"]["Arm 2"] = list(filter(lambda x: x.start == a2key[0], doc.ents))[0].text  # gets arm 2 entity

            # create dataframe from dictionary
            df = DataFrame.from_dict(final_dict, orient='columns', dtype=None, columns=None).transpose()
            df.index.name = 'Outcomes'
            df.reset_index(inplace=True)
            if pos_neg == 0:
                pos_dataframes.append(df)
            else:
                neg_dataframes.append(df)
            pos_neg = 1
    return pos_dataframes, neg_dataframes


def evaluate_tables(gold_path, predicted_path, negative):
    micro_prf = PRFScore()
    with open(gold_path, newline='') as goldcsv:
        gold_examples = [d for d in csv.DictReader(goldcsv) if d[''] != '']
    with open(predicted_path, newline='') as predcsv:
        positive_predictions = [d for d in csv.DictReader(predcsv) if d[''] != '']
    with open(negative, newline='') as negcsv:
        negative_predictions = [d for d in csv.DictReader(negcsv) if d[''] != '']
    tp = 0
    fp = 0
    fn = 0
    for ppred in positive_predictions:
        if ppred in gold_examples and ppred[''] != '':
            micro_prf.tp += 1
            tp += 1
        else:
            micro_prf.fp += 1
            fp += 1
    for npred in negative_predictions:
        if npred in gold_examples and ppred[''] != '':
            micro_prf.fn += 1
            fn += 1
    print("tp: ", tp," fp: ", fp," fn: ", fn)
    return {
        "rel_micro_p": micro_prf.precision,
        "rel_micro_r": micro_prf.recall,
        "rel_micro_f": micro_prf.fscore,
    }

 #       for grow,prow in zip(gold,pred):
  #          print(grow,prow)


#rel_processing(nlp2, Path)

if __name__ == "__main__":
    # instantiate pipeline inputs
    nlp = spacy.blank("en")
    doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
    docs = doc_bin.get_docs(nlp.vocab)

    #output_dfs, negatives = tabulate_pico_entities(nlp, docs) #  build gold tables

    # full pipeline with ner and rel models
    #ner_preds = named_entity_recognition(ner_model_path, docs)
    #rel_preds = relation_extraction(rel_model_path, ner_preds)
    #positives, negatives = tabulate_pico_entities(nlp, rel_preds)


    # output gold
    #with open('../datasets/preprocessed/all_domains/gold_table/all_domains_gold_table.csv', 'a') as output:
     #   for df in output_dfs:
      #      df.to_csv(output)
       #     output.write("\n")

    # output positives
    #with open('../output_tables/all_domains/positives_table.csv', 'a') as output:
     #   for df in positives:
      #      df.to_csv(output)
       #     output.write("\n")

    # output negatives
    #with open('../output_tables/all_domains/negatives_table.csv', 'a') as output:
     #   for df in negatives:
      #      df.to_csv(output)
       #     output.write("\n")

    #output.close()
   # print(evaluate_tables("../datasets/preprocessed/all_domains/gold_table/all_domains_gold_table.csv",
    #                '../output_tables/all_domains/positives_table.csv','../output_tables/all_domains/negatives_table.csv'))






#def score_relations(gold_dict, pred_dict):
    """Score a batch of examples."""
    micro_prf = PRFScore()

   # for example in examples:
    #    gold = example.reference._.rel
     #   pred = example.predicted._.rel
      #  for key, pred_dict in pred.items():
       #     gold_labels = [k for (k, v) in gold[key].items() if v == 1.0]
        #    for k, v in pred_dict.items():
         #       if v >= threshold:
          #          if k in gold_labels:
           #             micro_prf.tp += 1
            #        else:
             #           micro_prf.fp += 1
              #  else:
               #     if k in gold_labels:
                #        micro_prf.fn += 1

#for name, proc in nlp.pipeline:

 #   pred = proc(doc)
  #  for rel_dict in pred._.rel.items():
   #     if rel_dict[1]['A1_MEASURE'] > 0.05 or rel_dict[1]['A2_MEASURE']>0.05:
    #        print(f"spans: {[(e.start, e.text, e.label_) for e in pred.ents]}")
     #       print(rel_dict)


#main(nlp, Path, True)

