"""
Evaluate the five NLP tasks of the RCT-ART system: NER, RE, JOINT NER + RE,
TABULATION (STRICT), TABULATION (RELAXED). Also generate confusion matrices.
"""
import spacy, operator, csv, os
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.scorer import Scorer,PRFScore
from spacy.vocab import Vocab

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def ner_evaluate(ner_model_path,test_data):
    """ Evaluates NER scores of model on test data, can output to console and/or file"""
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
    """Evaluates joint performance of ner and rel extraction model,
    as well as the rel model alone if only rel model provided"""
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
            #print()
            #print(f"Text: {gold.text}")
            #print(f"gold_spans: {[(e.start, e.text, e.label_) for e in gold.ents]}")
            #print(f"pred_spans: {[(e.start, e.text, e.label_) for e in pred.ents]}")
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
    thresholds = [0.000, 0.050, 0.100, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    #print()
    #print("Results of the trained model:")
    task = False
    if ner_model_path != None:
        task = True
    _score_and_format(examples, thresholds, task)


def _score_and_format(examples, thresholds, task):
    """outputs rel and joint performance scores, to console and/or txt file"""
    if task:
        outfile.write("Joint\n")
    else:
        outfile.write("Rel alone\n")
    for threshold in thresholds:
        r = score_relations(examples, threshold)
        results = {k: "{:.2f}".format(v * 100) for k, v in r.items()}
        #print(f"threshold {'{:.2f}'.format(threshold)} \t {results}")
        outfile.write(f"threshold {'{:.2f}'.format(threshold)} \t {results}\n")


def evaluate_result_tables(gold_path, predicted_path, strict = True):
    """ Evaluates performance of model on tabulation task, compares prediction tables
    vs gold tables, can output to console and/or txt file"""
    print("|| Evaluating table task performance")
    prf = PRFScore()
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
            print("error")
            continue # empty lists in gold are error in data
        if pred_list == []: # empty lists in pred are false negatives if not empty in gold
            print("")
            for gold in gold_list:
                del gold['']
                examples.append({"gold": gold, "pred": {}})

    if strict: # assess table with exact entity matches
        for example in examples:
            if not example["pred"]:
                prf.fn += 1
                print("FALSE NEGATIVE ->", example)
            else:
                if example["pred"] == example["gold"]:
                    prf.tp += 1
                    print("TRUE POSTIVE ->", example)
                else:
                    prf.fp += 1
                    print("FALSE POSITIVE ->", example)s

    else: # assess tables with less strict entity criteria -- gold/pred entity boundary overlap
        for example in examples:
            relaxed_match = True
            if not example["pred"]: prf.fn += 1 # empty prediction --> false negative
            else:
                for pval, gval in zip(example["pred"].values(), example["gold"].values()):
                    if gval not in pval and pval not in gval:
                        relaxed_match = False
                if relaxed_match: prf.tp += 1
                else: prf.fp += 1

    output = {"rel_micro_p": prf.precision,
              "rel_micro_r": prf.recall,
              "rel_micro_f": prf.fscore,}
    outfile.write("Table_evaluation")
    if strict: outfile.write("strict\n")
    else: outfile.write("relaxed\n")
    outfile.write(f"{output}\n")
    print(output)


def create_ner_confusion_matrix(model_path, test_path):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    ner = spacy.load(model_path)
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    gold_docs = list(doc_bin.get_docs(ner.vocab))
    pred_docs = [ner(gold_doc.text) for gold_doc in gold_docs]
    gold_array = []
    pred_array = []
    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        for g_tok,p_tok in zip(gold_doc, pred_doc):
            if g_tok.ent_type_ == '':
                gold_array.append("NO_ENT")
            else:
                gold_array.append(g_tok.ent_type_)
            if p_tok.ent_type_ == '':
                pred_array.append("NO_ENT")
            else:
                pred_array.append(p_tok.ent_type_)
    cm = confusion_matrix(gold_array, pred_array,
                          labels=["OC","INTV","MEAS","NO_ENT"],
                          sample_weight=None, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["OC","INTV","MEAS","NO_ENT"])
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 24}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = disp.plot(include_values=True,
             cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()


def create_rel_confusion_matrix(model_path, test_path):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from scripts.tabulate import relation_extraction
    vocab = Vocab()
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    for_pred = list(doc_bin.get_docs(vocab))
    pred_docs = relation_extraction(model_path, for_pred)
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    gold_docs = list(doc_bin.get_docs(vocab))
    pred_array, pred_keys, gold_keys, gold_array = [], [], [], []
    for pred_doc, gold_doc in zip(pred_docs,gold_docs):
        for pkey, p_rel_dict in pred_doc._.rel.items():
            pred_keys.append(pkey)
            if pkey in gold_doc._.rel.keys():
                gold_keys.append(pkey)
                gold_rel = gold_doc._.rel[pkey]  # get relation
                max_gold = max(gold_rel.items(),
                              key=operator.itemgetter(1))  # selects highest probability relation
                if max_gold[1] > 0.5:  # includes relation if above set threshold for probability
                    gold_array.append(max_gold[0])
                else:
                    gold_array.append("NO_RE")
                pred_rel = pred_doc._.rel[pkey]  # get relation
                max_pred = max(pred_rel.items(),
                              key=operator.itemgetter(1))  # selects highest probability relation
                if max_pred[1] > 0.5:  # includes relation if above set threshold for probability
                    pred_array.append(max_pred[0])
                else:
                    pred_array.append("NO_RE")

    cm = confusion_matrix(gold_array, pred_array, labels=["A1_RES", "A2_RES", "OC_RES", "NO_RE"],
                          sample_weight=None, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A1_RES", "A2_RES", "OC_RES", "NO_RE"])
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 24}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()



if __name__ == "__main__":
    # some of these paths require trained models to be in place already
    doc_path = "../ddatasets/4_preprocessed/all_domains/test.spacy"
    gold_table_path = "../datasets/5_gold_tables/all_domains"
    pred_table_path = "../output_tables/all_domains_"
    model_bases = ["biobert","scibert","roberta"]
    #ner_evaluate(f"../trained_models/biobert/ner/all_domains/model-best", doc_path)

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
        evaluate_result_tables(gold_table_path, f"{pred_table_path}{model_base}", strict=True)
        # assess table relaxed performance
        evaluate_result_tables(gold_table_path, f"{pred_table_path}{model_base}", strict=False)

        outfile.close()

    #create_ner_confusion_matrix("../trained_models/ner/all_domains/model-best", doc_path)
    #create_rel_confusion_matrix("../trained_models/rel/all_domains/model-best", doc_path)

