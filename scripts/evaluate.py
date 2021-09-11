import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
import scripts.entity_ruler
import operator
import csv
from spacy.scorer import Scorer,PRFScore
import os
from collections import defaultdict


# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def ner_evaluate(ner_model_path,test_data):
    """ Evaluates NER scores of model on test data"""
    print("|| Loading model for NER task")
    nlp = spacy.load(ner_model_path)
    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    examples = []
    for gold in docs:
        pred = nlp(gold.text)
        examples.append(Example(pred, gold))
    print("|| Evaluating NER task performance")
    print(nlp.evaluate(examples))
    outfile.write("NER_evaluation\n")
    outfile.write(f"{nlp.evaluate(examples)}\n")


# This function was extensively adapted from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
# it can now be used to evaluate joint entity--relation extraction performance
def joint_ner_rel_evaluate(ner_model_path, rel_model_path, test_data, print_details: bool):
    """Evaluates joint performance of ner and rel extraction model, as well as the rel model alone
    if only rel model provided"""
    if ner_model_path != None:
        print("|| Loading models for joint task")
        ner = spacy.load(ner_model_path)
        print("|| Evaluating joint task performance")
    else:
        print("|| Loading models for rel task")
        print("|| Evaluating rel task performance")
    rel = spacy.load(rel_model_path)

    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(rel.vocab)
    examples = []
    for gold in docs:
        pred = Doc(
            rel.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        if ner_model_path != None:
            pred.ents = ner(gold.text).ents
        else:
            pred.ents = gold.ents
        for name, proc in rel.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, gold))

        # Print the gold and prediction, if gold label is not 0
        if print_details:
            print()
            print(f"Text: {gold.text}")
            print(f"gold_spans: {[(e.start, e.text, e.label_) for e in gold.ents]}")
            print(f"pred_spans: {[(e.start, e.text, e.label_) for e in pred.ents]}")
            gold_ents = [e.text for e in gold.ents]
            assessed_ents = []
            for value, rel_dict in pred._.rel.items():
                try:
                    gold_labels = [k for (k, v) in gold._.rel[value].items() if v == 1.0]
                    if gold_labels:
                        print(
                            f" pair: {value} --> gold labels: {gold_labels} --> predicted values: {rel_dict}"
                        )
                except KeyError:
                    pred_rel = max(rel_dict.items(),key=operator.itemgetter(1))
                    if pred_rel[1] > 0.5:
                        print("Relation mapped with wrong entity pair")
                    else:
                        parent_ent = list(filter(lambda x: x.start == value[0], pred.ents))[0].text
                        child_ent = list(filter(lambda x: x.start == value[1], pred.ents))[0].text
                        if parent_ent not in assessed_ents:
                            if parent_ent in gold_ents:
                                print(parent_ent," Correct entity and correctly didn't map relation")
                            else:
                                print(parent_ent," incorrect entity")
                            assessed_ents.append(parent_ent)
                        if child_ent not in assessed_ents:
                            if child_ent in gold_ents:
                                print(child_ent, "Correct entity and correctly didn't map relation")
                            else:
                                print(child_ent, "incorrect entity")
                            assessed_ents.append(child_ent)
            print()
    random_examples = []
    docs = doc_bin.get_docs(rel.vocab)
    for gold in docs:
        pred = Doc(
            rel.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        if ner_model_path != None:
            pred.ents = ner(gold.text).ents
        else:
            pred.ents = gold.ents
        relation_extractor = rel.get_pipe("relation_extractor")
        get_instances = relation_extractor.model.attrs["get_instances"]
        for (e1, e2) in get_instances(pred):
            offset = (e1.start, e2.start)
            if offset not in pred._.rel:
                pred._.rel[offset] = {}
            for label in relation_extractor.labels:
                pred._.rel[offset][label] = random.uniform(0, 1)
        random_examples.append(Example(pred, gold))

    thresholds = [0.000, 0.050, 0.100, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    #print()
    #print("Random baseline:")
    #if ner_model_path != None:
     #   task = True
    #_score_and_format(examples, thresholds, task)

    print()
    print("Results of the trained model:")
    #outfile.write()
    task = False
    if ner_model_path != None:
        task = True
    _score_and_format(examples, thresholds, task)


def _score_and_format(examples, thresholds, task):
    if task:
        outfile.write("Joint\n")
    else:
        outfile.write("Rel alone\n")
    for threshold in thresholds:
        r = score_relations(examples, threshold)
        results = {k: "{:.2f}".format(v * 100) for k, v in r.items()}
        print(f"threshold {'{:.2f}'.format(threshold)} \t {results}")
        outfile.write(f"threshold {'{:.2f}'.format(threshold)} \t {results}\n")


def evaluate_result_tables(gold_path, predicted_path, strict = True):
    """ Evaluates performance of model on tabulation task, compares prediction tables vs gold tables"""
    print("|| Evaluating table task performance")
    micro_prf = PRFScore()
    examples = []
    for gold_csv, pred_csv in zip(os.listdir(gold_path), os.listdir(predicted_path)):
        gold_open = open(os.path.join(gold_path, gold_csv), newline='')
        pred_open = open(os.path.join(predicted_path, pred_csv), newline='')
        gold_list = [d for d in csv.DictReader(gold_open)]
        pred_list = [d for d in csv.DictReader(pred_open)]
        for gold, pred in zip(gold_list,pred_list):
            del gold['']
            del pred['']
            examples.append({"gold":gold,"pred":pred})
            continue
        if gold_list == []:
            continue # empty lists in gold are error in data and should be skipped
        if pred_list == []: # empty lists in pred are false negatives if not empty in gold
            for gold in gold_list:
                del gold['']
                examples.append({"gold": gold, "pred": {}})

    if strict: # assess table with exact entity matches
        for example in examples:
            if not example["pred"]: micro_prf.fn += 1
            else:
                if example["pred"] == example["gold"]: micro_prf.tp += 1
                else: micro_prf.fp += 1

    else: # assess tables with less strict entity criteria, checking if pred entity is within gold entity boundary
        for example in examples:
            relaxed_match = True
            if not example["pred"]: micro_prf.fn += 1 # empty prediction for existing gold table, false negative
            else:
                for pval, gval in zip(example["pred"].values(), example["gold"].values()):
                    if pval not in gval and example["pred"]:
                        relaxed_match = False
                if relaxed_match: micro_prf.tp += 1
                else: micro_prf.fp += 1

    output = {"rel_micro_p": micro_prf.precision,
              "rel_micro_r": micro_prf.recall,
              "rel_micro_f": micro_prf.fscore,}
    outfile.write("Table_evaluation")
    if strict: outfile.write("strict\n")
    else: outfile.write("relaxed\n")
    outfile.write(f"{output}\n")
    print(output)


if __name__ == "__main__":
    #typer.run(ner_rel_evaluate)
    file_name = "BERT_baselines"
    outfile = open(f"../evaluation_results/{file_name}.txt", "w")
    doc_path = "../datasets/preprocessed/all_domains/results_only/test.spacy"
    rel_model_path = "../trained_models/rel/all_domains/model-best"
    gold_table_path = "../datasets/preprocessed/all_domains/gold_tables"
    pred_table_path = "../output_tables"
    model_bases = ["biobert","scibert","roberta"]

    # evaluate different model-bases
    for model_base in model_bases:
        outfile = open(f"../evaluation_results/{model_base}.txt", "w")
        # assess ner performance
        ner_evaluate(f"../trained_models/{model_base}/ner/all_domains/model-best",doc_path)
        # assess rel performance
        joint_ner_rel_evaluate(None,f"../trained_models/{model_base}/rel/all_domains/model-best",doc_path,False)
        # assess joint performance
        joint_ner_rel_evaluate(f"../trained_models/{model_base}/ner/all_domains/model-best"
                               ,f"../trained_models/{model_base}/rel/all_domains/model-best",doc_path,False)
        # assess table strict performance
        evaluate_result_tables(gold_table_path, pred_table_path, strict=True)
        # assess table relaxed performance
        evaluate_result_tables(gold_table_path, pred_table_path, strict=False)

        outfile.close()