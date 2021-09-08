import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
import scripts.entity_ruler
import operator

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# This function was adapted from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
# it can now be used to evaluate joint entity--relation extraction performance
def ner_rel_evaluate(ner_model_path, rel_model_path, test_data, print_details: bool):
    """Evaluates joint performance of ner and rel extraction model, as well as the rel model alone
    if gold entities were provided"""
    print("|| Loading models")
    if ner_model_path != None:
        ner = spacy.load(ner_model_path)
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
    print("|| Getting model scores")
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
    print()
    print("Random baseline:")
    _score_and_format(random_examples, thresholds)

    print()
    print("Results of the trained model:")
    _score_and_format(examples, thresholds)


def _score_and_format(examples, thresholds):
    for threshold in thresholds:
        r = score_relations(examples, threshold)
        results = {k: "{:.2f}".format(v * 100) for k, v in r.items()}
        print(f"threshold {'{:.2f}'.format(threshold)} \t {results}")


if __name__ == "__main__":
    #typer.run(ner_rel_evaluate)
    doc_path = "../datasets/preprocessed/all_domains/results_only/test.spacy"
    ner_model_path = "../trained_model/ner/all_domains/model-best"
    rel_model_path = "../trained_model/rel_pipeline/all_domains/model-best"
    ner_rel_evaluate(ner_model_path,rel_model_path,doc_path,True)
