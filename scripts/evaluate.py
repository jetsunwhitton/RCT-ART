"""
Evaluate the five NLP tasks of the RCT-ART system: NER, RE, JOINT NER + RE,
TABULATION (STRICT), TABULATION (RELAXED). Also generate confusion matrices.
"""
import spacy, operator, csv, os
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.scorer import Scorer,PRFScore
from spacy.vocab import Vocab
from itertools import zip_longest
import pandas as pd

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
    return nlp.evaluate(examples)



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
    task = False
    if ner_model_path != None:
        task = True
    return _score_and_format(examples, thresholds, task)


def _score_and_format(examples, thresholds, task):
    """outputs rel and joint performance scores, to console and/or txt file"""
    threshold_dict = {}
    for threshold in thresholds:
        r = score_relations(examples, threshold)
        threshold_dict[threshold] = {k: "{:.2f}".format(v) for k, v in r.items()}
    return threshold_dict


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
        for gold, pred in zip_longest(gold_list,pred_list,fillvalue=False):
            if gold != False: del gold[''] # remove extra CSV formatting
            if pred != False: del pred['']

            # set pred to false for false negative if intv row empty when gold isn't
            if pred == {'outcome': 'intervention', 'arm 1': '', 'arm 2': ''} \
                    and gold != {'outcome': 'intervention', 'arm 1': '', 'arm 2': ''}: pred = False

            examples.append({"gold":gold,"pred":pred})
        if gold_list == []:
            print("error")
            continue # empty lists in gold are error in data az
        if pred_list == []: # empty lists in pred are false negatives if not empty in gold
            for gold in gold_list:
                del gold['']
                examples.append({"gold": gold, "pred": {}})

    if strict: # assess table with exact entity matches
        for example in examples:
            if not example["pred"]:
                prf.fn += 1
                print("FN ----> ", example)
            elif not example["gold"]:
                prf.fp += 1
            else:
                if example["pred"] == example["gold"]:
                    prf.tp += 1
                    print("TP ----> ", example)
                else:
                    prf.fp += 1
                    print("FP ----> ", example)

    else: # assess tables with less strict entity criteria -- gold/pred entity boundary overlap
        for example in examples:
            relaxed_match = True
            if not example["pred"]: prf.fn += 1 # empty prediction --> false negative
            elif not example["gold"]: prf.fp += 1 # prediction made when no gold tuple --> false postive
            else:
                for pval, gval in zip(example["pred"].values(), example["gold"].values()):
                    if gval not in pval and pval not in gval:
                        relaxed_match = False
                if relaxed_match: prf.tp += 1
                else: prf.fp += 1

    output = {"rel_micro_p": prf.precision,
              "rel_micro_r": prf.recall,
              "rel_micro_f": prf.fscore,}
    print(output)
    return output


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
            'size': 16}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(16, 16))
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
            'size': 16}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()


def save(data, path):
    """Save infomation to CSV database file"""
    try:
        os.mkdir(path)
    except:
        pass
    #save runs
    runs_file = path + '/runs.csv'
    df = pd.DataFrame.from_records(data)
    df.to_csv(runs_file)

    #save mean and SD
    means_stds_file = path + '/means_stds.csv'
    means_stds = pd.concat([df.mean(),df.std()],axis=1,keys=['Mean', 'Std'])
    means_stds.to_csv(means_stds_file)


def eval_base_models(doc_path, model_bases, seed_config_path, gold_table_path, pred_table_path,output):
    for model_base in model_bases:
        model_overview_runs = []
        model_ner_label_runs = []
        model_rel_label_runs = []
        for seed_run in os.listdir(seed_config_path):
            models_output = {"NER_P": [], "NER_R": [], "NER_F": [],
                             "REL_P": [], "REL_R": [], "REL_F": [],
                             "JOINT_P": [], "JOINT_R": [], "JOINT_F": [],
                             "STRICT_P": [], "STRICT_R": [], "STRICT_F": [],
                             "RELAXED_P": [], "RELAXED_R": [], "RELAXED_F": []}

            model_ner_labels = {"OC_P": [], "OC_R": [], "OC_F": [],
                                "INTV_P": [], "INTV_R": [], "INTV_F": [],
                                "MEAS_P": [], "MEAS_R": [], "MEAS_F": []}

            model_rel_labels = {"A1_RES_P": [], "A1_RES_R": [], "A1_RES_F": [],
                                "A2_RES_P": [], "A2_RES_R": [], "A2_RES_F": [],
                                "OC_RES_P": [], "OC_RES_R": [], "OC_RES_F": []}

            """assess ner performance"""
            # overall performance
            ner_res = ner_evaluate(f"../trained_models/{model_base}/ner/all_domains/{seed_run}/model-best", doc_path)

            models_output["NER_P"], models_output["NER_R"], models_output["NER_F"] = \
                ner_res['ents_p'], ner_res['ents_r'], ner_res['ents_f']

            # label performance
            ner_label_res = ner_res['ents_per_type']

            oc_res = ner_label_res['OC']
            model_ner_labels["OC_P"], model_ner_labels["OC_R"], model_ner_labels["OC_F"] = oc_res['p'], oc_res['r'], \
                                                                                           oc_res['f']

            intv_res = ner_label_res['INTV']
            model_ner_labels["INTV_P"], model_ner_labels["INTV_R"], model_ner_labels["INTV_F"] = intv_res['p'], \
                                                                                                 intv_res['r'], \
                                                                                                 intv_res['f']

            meas_res = ner_label_res['MEAS']
            model_ner_labels["MEAS_P"], model_ner_labels["MEAS_R"], model_ner_labels["MEAS_F"] = meas_res['p'], \
                                                                                                 meas_res['r'], \
                                                                                                 meas_res['f']

            model_ner_label_runs.append(model_ner_labels)

            """assess rel performance"""
            # overall performance
            rel_res = joint_ner_rel_evaluate(None,
                                             f"../trained_models/{model_base}/rel/all_domains/{seed_run}/model-best",
                                             doc_path, False)
            rel_at_treshold = rel_res[0.5]
            models_output["REL_P"], models_output["REL_R"], models_output["REL_F"] = \
                float(rel_at_treshold['rel_micro_p']), float(rel_at_treshold['rel_micro_r']), float(
                    rel_at_treshold['rel_micro_f'])

            # label performance
            model_rel_labels["A1_RES_P"], model_rel_labels["A1_RES_R"], model_rel_labels["A1_RES_F"], \
            model_rel_labels["A2_RES_P"], model_rel_labels["A2_RES_R"], model_rel_labels["A2_RES_F"], \
            model_rel_labels["OC_RES_P"], model_rel_labels["OC_RES_R"], model_rel_labels["OC_RES_F"] = \
                float(rel_at_treshold['a1_res_p']), float(rel_at_treshold['a1_res_r']), float(
                    rel_at_treshold['a1_res_f']), \
                float(rel_at_treshold['a2_res_p']), float(rel_at_treshold['a2_res_r']), float(
                    rel_at_treshold['a2_res_f']), \
                float(rel_at_treshold['oc_res_p']), float(rel_at_treshold['oc_res_r']), float(
                    rel_at_treshold['oc_res_f'])

            model_rel_label_runs.append(model_rel_labels)

            """assess joint performance"""
            joint_res = joint_ner_rel_evaluate(f"../trained_models/{model_base}/ner/all_domains/{seed_run}/model-best"
                                               ,
                                               f"../trained_models/{model_base}/rel/all_domains/{seed_run}/model-best",
                                               doc_path, False)
            joint_at_treshold = joint_res[0.5]
            models_output["JOINT_P"], models_output["JOINT_R"], models_output["JOINT_F"] = \
                float(joint_at_treshold['rel_micro_p']), float(joint_at_treshold['rel_micro_r']), float(
                    joint_at_treshold['rel_micro_f'])

            """assess table strict performance"""
            strict_res = evaluate_result_tables(gold_table_path, f"{pred_table_path}{model_base}/{seed_run}",
                                                strict=True)
            models_output["STRICT_P"], models_output["STRICT_R"], models_output["STRICT_F"] = \
                strict_res['rel_micro_p'], strict_res['rel_micro_r'], strict_res['rel_micro_f']

            """assess table relaxed performance"""
            relaxed_res = evaluate_result_tables(gold_table_path, f"{pred_table_path}{model_base}/{seed_run}",
                                                 strict=False)
            models_output["RELAXED_P"], models_output["RELAXED_R"], models_output["RELAXED_F"] = \
                relaxed_res['rel_micro_p'], relaxed_res['rel_micro_r'], relaxed_res['rel_micro_f']
            model_overview_runs.append(models_output)
        save(model_overview_runs,
             f"{output}/{model_base}")
        save(model_ner_label_runs,
             f"{output}/{model_base}/labels/ner")
        save(model_rel_label_runs,
             f"{output}/{model_base}/labels/rel")


def eval_strat_models(doc_path, ner_model_strats, seed_config_path, gold_table_path, output):
    for strat in os.listdir(ner_model_strats):
        model_strat_runs = []
        for seed_run in os.listdir(seed_config_path):
            models_output = {"NER_P": [], "NER_R": [], "NER_F": [],
                             "REL_P": [], "REL_R": [], "REL_F": [],
                             "JOINT_P": [], "JOINT_R": [], "JOINT_F": [],
                             "STRICT_P": [], "STRICT_R": [], "STRICT_F": [],
                             "RELAXED_P": [], "RELAXED_R": [], "RELAXED_F": []}

            """assess ner performance"""
            # overall performance
            ner_res = ner_evaluate(f"../trained_models/biobert/ner/all_domain_strats/{strat}/{seed_run}/model-best", doc_path)

            models_output["NER_P"], models_output["NER_R"], models_output["NER_F"] = \
                ner_res['ents_p'], ner_res['ents_r'], ner_res['ents_f']

            """assess rel performance"""
            # overall performance
            rel_res = joint_ner_rel_evaluate(None,
                                             f"../trained_models/biobert/rel/all_domain_strats/{strat}/{seed_run}/model-best",
                                             doc_path, False)
            rel_at_treshold = rel_res[0.5]
            models_output["REL_P"], models_output["REL_R"], models_output["REL_F"] = \
                float(rel_at_treshold['rel_micro_p']), float(rel_at_treshold['rel_micro_r']), float(
                    rel_at_treshold['rel_micro_f'])


            """assess joint performance"""
            joint_res = joint_ner_rel_evaluate(f"../trained_models/biobert/ner/all_domain_strats/{strat}/{seed_run}/model-best",
                                               f"../trained_models/biobert/rel/all_domain_strats/{strat}/{seed_run}/model-best",
                                               doc_path, False)
            joint_at_treshold = joint_res[0.5]
            models_output["JOINT_P"], models_output["JOINT_R"], models_output["JOINT_F"] = \
                float(joint_at_treshold['rel_micro_p']), float(joint_at_treshold['rel_micro_r']), float(
                    joint_at_treshold['rel_micro_f'])

            """assess table strict performance"""
            strict_res = evaluate_result_tables(gold_table_path, f"../output_tables/all_domains_strats/train_{strat}/{seed_run}",
                                                strict=True)
            models_output["STRICT_P"], models_output["STRICT_R"], models_output["STRICT_F"] = \
                strict_res['rel_micro_p'], strict_res['rel_micro_r'], strict_res['rel_micro_f']

            """assess table relaxed performance"""
            relaxed_res = evaluate_result_tables(gold_table_path, f"../output_tables/all_domains_strats/train_{strat}/{seed_run}",
                                                 strict=False)
            models_output["RELAXED_P"], models_output["RELAXED_R"], models_output["RELAXED_F"] = \
                relaxed_res['rel_micro_p'], relaxed_res['rel_micro_r'], relaxed_res['rel_micro_f']

            model_strat_runs.append(models_output)

        save(model_strat_runs,
             f"{output}/{strat}")


def eval_domain_models(domain_cuts, seed_config_path, gold_table_path, output):
    for domain in os.listdir(f"../datasets/4_preprocessed/{domain_cuts}"):
        print(domain)
        model_domain_runs = []
        doc_path = f"../datasets/4_preprocessed/{domain_cuts}/{domain}/test.spacy"
        for seed_run in os.listdir(seed_config_path):
            print(seed_run)
            models_output = {"NER_P": [], "NER_R": [], "NER_F": [],
                             "REL_P": [], "REL_R": [], "REL_F": [],
                             "JOINT_P": [], "JOINT_R": [], "JOINT_F": [],
                             "STRICT_P": [], "STRICT_R": [], "STRICT_F": [],
                             "RELAXED_P": [], "RELAXED_R": [], "RELAXED_F": []}

            """assess ner performance"""
            # overall performance
            ner_res = ner_evaluate(f"../trained_models/biobert/ner/{domain_cuts}/{domain}/{seed_run}/model-best", doc_path)

            models_output["NER_P"], models_output["NER_R"], models_output["NER_F"] = \
                ner_res['ents_p'], ner_res['ents_r'], ner_res['ents_f']

            """assess rel performance"""
            # overall performance
            rel_res = joint_ner_rel_evaluate(None,
                                             f"../trained_models/biobert/rel/{domain_cuts}/{domain}/{seed_run}/model-best",
                                             doc_path, False)
            rel_at_treshold = rel_res[0.5]
            models_output["REL_P"], models_output["REL_R"], models_output["REL_F"] = \
                float(rel_at_treshold['rel_micro_p']), float(rel_at_treshold['rel_micro_r']), float(
                    rel_at_treshold['rel_micro_f'])


            """assess joint performance"""
            joint_res = joint_ner_rel_evaluate(f"../trained_models/biobert/ner/{domain_cuts}/{domain}/{seed_run}/model-best",
                                               f"../trained_models/biobert/rel/{domain_cuts}/{domain}/{seed_run}/model-best",
                                               doc_path, False)
            joint_at_treshold = joint_res[0.5]
            models_output["JOINT_P"], models_output["JOINT_R"], models_output["JOINT_F"] = \
                float(joint_at_treshold['rel_micro_p']), float(joint_at_treshold['rel_micro_r']), float(
                    joint_at_treshold['rel_micro_f'])

            """assess table strict performance"""
            strict_res = evaluate_result_tables(f"{gold_table_path}/{domain_cuts}/{domain}", f"../output_tables/{domain_cuts}/{domain}/{seed_run}",
                                                strict=True)
            models_output["STRICT_P"], models_output["STRICT_R"], models_output["STRICT_F"] = \
                strict_res['rel_micro_p'], strict_res['rel_micro_r'], strict_res['rel_micro_f']

            """assess table relaxed performance"""
            relaxed_res = evaluate_result_tables(f"{gold_table_path}/{domain_cuts}/{domain}", f"../output_tables/{domain_cuts}/{domain}/{seed_run}",
                                                 strict=False)
            models_output["RELAXED_P"], models_output["RELAXED_R"], models_output["RELAXED_F"] = \
                relaxed_res['rel_micro_p'], relaxed_res['rel_micro_r'], relaxed_res['rel_micro_f']

            model_domain_runs.append(models_output)

        save(model_domain_runs,
             f"{output}/{domain_cuts}/{domain}")

if __name__ == "__main__":

    # some of these paths require trained models to be in place already
    #doc_path = "../datasets/4_preprocessed/all_domains/test.spacy"
    #gold_table_path = "../datasets/5_gold_tables/all_domains"
    #pred_table_path = "../output_tables/all_domains_"
    #model_bases = ["biobert", "scibert","roberta"]
    #output = "../evaluation_results/models_all_domains"
    #seed_config_path = "../configs/ner/biobert"

    # evaluate different model-bases
    #eval_base_models(model_bases,seed_config_path,doc_path,gold_table_path,pred_table_path)

    # evaluate different training size strats

    ner_model_strats = "../trained_models/biobert/ner/all_domain_strats"
    seed_config_path = "../configs/ner/biobert"
    pred_table_path = "../output_tables/all_domains_strats/train_"
    output = "../evaluation_results/strats_all_domains"

    #eval_strat_models(doc_path, ner_model_strats, seed_config_path, gold_table_path, output)

    # evaluate different training size strats
    #doc_path = "../datasets/4_preprocessed/all_domains/test.spacy"
    #ner_model_strats = "../trained_models/biobert/ner/all_domain_strats"
    #seed_config_path = "../configs/ner/biobert"
    #gold_table_path = "../datasets/5_gold_tables/all_domains"
    #pred_table_path = "../output_tables/all_domains_strats/train_"
    domaincuts = ["out_of_domain", "capped_for_comparison", "capped_mix"]
    output = "../evaluation_results"
    for domaincut in domaincuts:
        print(domaincut)
        gold_table_path = f"../datasets/5_gold_tables"
        eval_domain_models(domaincut, seed_config_path, gold_table_path, output)

    # evaluate out of domain perfomance
    #for domain in os.listdir("../datasets/4_preprocessed/out_of_domain"):
     #   outfile = open(f"../evaluation_results/{domain}.txt", "w")
      #  print(domain)
       # doc_path = f"../datasets/4_preprocessed/out_of_domain/{domain}/test.spacy"
       # ner_model = f"../trained_models/biobert/ner/out_of_domain/{domain}/model-best"
       # rel_model = f"../trained_models/biobert/rel/out_of_domain/{domain}/model-best"
       # gold_table_path = f"../datasets/5_gold_tables/out_of_domain/{domain}"
       # pred_table_path = f"../output_tables/out_of_domain/{domain}"
        # assess ner performance
       # ner_evaluate(ner_model, doc_path)
        # assess rel performance
       # joint_ner_rel_evaluate(None, rel_model, doc_path, False)
        # assess joint performance
       # joint_ner_rel_evaluate(ner_model, rel_model, doc_path, False)
        # assess table strict performance
        #evaluate_result_tables(gold_table_path, pred_table_path, strict=True)
        # assess table relaxed performance
        #evaluate_result_tables(gold_table_path, pred_table_path, strict=False)

        #outfile.close()

     # evaluate single capped domain perfomance
     #for domain in os.listdir("../datasets/4_preprocessed/capped_for_comparison"):
        #outfile = open(f"../evaluation_results/{domain}.txt", "w")
        #print("single ",domain)
        #doc_path = f"../datasets/4_preprocessed/capped_for_comparison/{domain}/test.spacy"
        #ner_model = f"../trained_models/biobert/ner/capped_for_comparison/{domain}/model-best"
        #rel_model = f"../trained_models/biobert/rel/capped_for_comparison/{domain}/model-best"
        #gold_table_path = f"../datasets/5_gold_tables/capped_for_comparison/{domain}"
        #pred_table_path = f"../output_tables/capped_for_comparison/{domain}"
        # assess ner performance
        #ner_evaluate(ner_model, doc_path)
        # assess rel performance
        #joint_ner_rel_evaluate(None, rel_model, doc_path, False)
        # assess joint performance
        #joint_ner_rel_evaluate(ner_model, rel_model, doc_path, False)
        # assess table strict performance
        #evaluate_result_tables(gold_table_path, pred_table_path, strict=True)
        # assess table relaxed performance
        #evaluate_result_tables(gold_table_path, pred_table_path, strict=False)

        #outfile.close()

     # evaluate single capped domain perfomance
     #for domain in os.listdir("../datasets/4_preprocessed/capped_mix"):
         #outfile = open(f"../evaluation_results/{domain}.txt", "w")
         #print("mixed", domain)
         #doc_path = f"../datasets/4_preprocessed/capped_mix/{domain}/test.spacy"
         #ner_model = f"../trained_models/biobert/ner/capped_mix/{domain}/model-best"
         #rel_model = f"../trained_models/biobert/rel/capped_mix/{domain}/model-best"
         #gold_table_path = f"../datasets/5_gold_tables/capped_mix/{domain}"
         #pred_table_path = f"../output_tables/capped_mix/{domain}"
         # assess ner performance
         #ner_evaluate(ner_model, doc_path)
         # assess rel performance
         #joint_ner_rel_evaluate(None, rel_model, doc_path, False)
         # assess joint performance
         #joint_ner_rel_evaluate(ner_model, rel_model, doc_path, False)
         # assess table strict performance
         #evaluate_result_tables(gold_table_path, pred_table_path, strict=True)
         # assess table relaxed performance
         #evaluate_result_tables(gold_table_path, pred_table_path, strict=False)

         #outf#ile.close()

    #create_ner_confusion_matrix("../trained_models/biobert/ner/all_domains/model-best", doc_path)
    #create_rel_confusion_matrix("../trained_models/biobert/rel/all_domains/model-best", doc_path)

