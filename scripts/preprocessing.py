"""
This preprocesing script parses annotated relations and entities from
the prodigy jsonln output files. Also includes splitting functions.
"""
import json, random, os, ast
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

def merge_jsonl(jsonl_dirs, output_path):
    """Merges gold standard JSONL files from different disease area domains"""
    merge_list = []
    for path in jsonl_dirs:
        loaded = open(path,"r").read()
        merge_list += loaded.split("\n")
    with open(output_path, "w") as output:
        for dict in merge_list:
            if dict == "\n":
                pass
            else:
                try:
                    dict = json.loads(dict)
                    output.write(json.dumps(dict) + "\n")
                except:
                    print(dict)
        output.close()


# This function was adapted from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
def annotations_to_spacy(json_loc):
    """Converts Prodigy annotations into doc object with custom rel attribute."""
    msg = Printer()
    MAP_LABELS = {
        "A1_RES": "A1_RES",
        "A2_RES": "A2_RES",
        "OC_RES": "OC_RES",
    }
    try:
        Doc.set_extension("rel", default={})
    except ValueError:
        print("Rel extension already set on doc")
    vocab = Vocab()
    docs = []

    with open(json_loc, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the PICO entities
                    spans = example["spans"]
                    entities = []
                    span_end_to_start = {}
                    for span in spans:
                        entity = doc.char_span(
                            span["start"], span["end"], label=span["label"]
                        )
                        span_end_to_start[span["token_end"]] = span["token_start"]
                        entities.append(entity)
                        span_starts.add(span["token_start"])
                    doc.ents = entities

                    # Parse the PICO relations
                    rels = {}
                    for x1 in span_starts:
                        for x2 in span_starts:
                            rels[(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        # swaps tokens to correct relation positions
                        start = span_end_to_start[relation["head"]]
                        end = span_end_to_start[relation["child"]]
                        label = relation["label"]
                        label = MAP_LABELS[label]
                        if label not in rels[(start, end)]:
                            rels[(start, end)][label] = 1.0


                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in MAP_LABELS.values():
                                if label not in rels[(x1, x2)]:
                                    rels[(x1, x2)][label] = 0.0
                    doc._.rel = rels
                    try:
                        pmid = ast.literal_eval(example["user_data"])
                        doc.user_data["pmid"] = pmid["pmid"]
                    except KeyError:
                        pass # pmids have not been added to glaucoma dataset
                    docs.append(doc)
                except KeyError as e:
                    msg.fail(f"Skipping doc because of key error")
    print(len(docs))
    return docs


def train_dev_test_split(docs,output_dir):
    """
    Splits spaCy docs collection into train test and dev datasets based on pmid
    :param docs: list
    :return: train dev and test spacy files for model training and testing
    """
    pmid_set = set()
    with_pmid = []
    without_pmid = []
    for doc in docs:
        try:
            pmid_set.add(doc.user_data["pmid"])
            with_pmid.append(doc)
        except KeyError: #  primarily for glaucoma dataset in instance of this study
            without_pmid.append(doc)
    pmid_list = list(pmid_set)
    random.shuffle(pmid_list)  # randomise pmids before train, dev, test split
    l = len(pmid_list)
    train = pmid_list[0:int(l * 0.7)]
    dev = pmid_list[int(l * 0.7):int(l * 0.8)]

    w_train, w_dev, w_test = [],[],[]
    for doc in with_pmid:
        if doc.user_data["pmid"] in train:
            w_train.append(doc)
        elif  doc.user_data["pmid"] in dev:
            w_dev.append(doc)
        else:
            w_test.append(doc)

    random.shuffle(without_pmid)  # randomise sentences without pubmed ids for dividing across sets
    l = len(without_pmid)
    wo_train = without_pmid[0:int(l * 0.7)]
    wo_dev = without_pmid[int(l * 0.7):int(l * 0.8)]
    wo_test = without_pmid[int(l * 0.8):]

    joined_train = w_train + wo_train
    joined_dev = w_dev + wo_dev
    joined_test = w_test + wo_test

    docbin = DocBin(docs=joined_train, store_user_data=True)
    docbin.to_disk(f"{output_dir}/train.spacy")
    print(f"{len(joined_train)} training sentences")

    docbin = DocBin(docs=joined_dev, store_user_data=True)
    docbin.to_disk(f"{output_dir}/dev.spacy")
    print(f"{len(joined_dev)} dev sentences")

    docbin = DocBin(docs=joined_test, store_user_data=True)
    docbin.to_disk(f"{output_dir}/test.spacy")
    print(f"{len(joined_test)} test sentences")


def stratify_train_examples(doc_path, strats):
    """Stratifies input docs and binaries them into new spacy files"""
    vocab = Vocab()
    doc_bin = DocBin(store_user_data=True).from_disk(doc_path)
    docs = list(doc_bin.get_docs(vocab))
    l = len(docs)
    for strat in strats:
        if str(strat)[-2] != "0":
            name = str(strat)[-1] + "0%"
        else:
            name = str(strat)[-1] + "%"
        doc_strat = docs[:int(l*strat)]
        docbin = DocBin(docs=doc_strat, store_user_data=True)
        docbin.to_disk(f"../datasets/preprocessed/all_domains/training_stratifications/"
                       f"train_strat_{name}.spacy")


def out_of_domain_split(doc_dirs, exclude):
    """excludes one domain from full domain train and dev sets for use as test set"""
    merged_docs = []
    vocab = Vocab()
    for dir in doc_dirs:
        for files in os.listdir(dir):
            doc_bin = DocBin(store_user_data=True).from_disk(f"{dir}/{files}")
            merged_docs += list(doc_bin.get_docs(vocab))
    l = len(merged_docs)
    train = merged_docs[0:int(l * 0.9)]
    dev = merged_docs[int(l * 0.9):]

    test = []
    test_dir = f"../datasets/preprocessed/{exclude}/results_only"
    for files in os.listdir(test_dir):
        doc_bin = DocBin(store_user_data=True).from_disk(f"{test_dir}/{files}")
        test += list(doc_bin.get_docs(vocab))

    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(f"../datasets/preprocessed/out_of_domain/{exclude}_as_test/train.spacy")
    print(f"{len(train)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(f"../datasets/preprocessed/out_of_domain/{exclude}_as_test/dev.spacy")
    print(f"{len(dev)} dev sentences")

    docbin = DocBin(docs=test, store_user_data=True)
    docbin.to_disk(f"../datasets/preprocessed/out_of_domain/{exclude}_as_test/test.spacy")
    print(f"{len(test)} test sentences")


def cap_docs(doc_dirs, names, cap):
    """Caps number of examples in datasets for comparison, outputting both individual
    sets and merged sets for incremental evaluation"""
    vocab = Vocab()
    merged_docs = []
    merged_names = ""
    count = 0
    for docs, name in zip(doc_dirs, names):
        doc_bin = DocBin(store_user_data=True).from_disk(docs)
        capped = list(doc_bin.get_docs(vocab))[:cap-1]
        random.shuffle(capped)
        l = len(capped)
        train = capped[:int(l*0.7)]
        dev = capped[int(l*0.8):int(l*0.9)]
        test = capped[int(l*0.9):]
        capped_train = DocBin(docs=train, store_user_data=True)
        capped_train.to_disk(f"../datasets/preprocessed/capped_for_comparison/{name}/train.spacy")
        capped_dev = DocBin(docs=dev, store_user_data=True)
        capped_dev.to_disk(f"../datasets/preprocessed/capped_for_comparison/{name}/dev.spacy")
        capped_test = DocBin(docs=test, store_user_data=True)
        capped_test.to_disk(f"../datasets/preprocessed/capped_for_comparison/{name}/test.spacy")

        # output merged sets of capped docs for incremental domain increase evaluation
        merged_docs += capped
        merged_names += name + "_"
        if count > 0:
            random.shuffle(merged_docs)
            train = merged_docs[:int(len(merged_docs)*0.9)]
            dev = merged_docs[int(len(merged_docs)*0.9):]
            merged_train = DocBin(docs= train, store_user_data=True)
            merged_train.to_disk(f"../datasets/preprocessed/capped_mix/{merged_names[:-1]}/train.spacy")
            merged_dev = DocBin(docs=dev, store_user_data=True)
            merged_dev.to_disk(f"../datasets/preprocessed/capped_mix/{merged_names[:-1]}/dev.spacy")
        count += 1


if __name__ == "__main__":
    #gold_dir = "../datasets/gold_result_annotations"
    #cardiovascular = f"{gold_dir}/cardiovascular_disease/cardiovascular_disease_gold.jsonl"
    #glaucoma = f"{gold_dir}/glaucoma/glaucoma_gold.jsonl"
    #merge_all_list = [f'../datasets/gold_result_annotations/{domain}/{domain}_gold.jsonl'
     #               for domain in os.listdir("../datasets/gold_result_annotations")
      #                if domain != "all_domains"]

    #stratify_train_examples("../datasets/preprocessed/all_domains/results_only/train.spacy",[0.05,0.5])
    #merge_jsonl(merge_all_list, "../datasets/gold_result_annotations/all_domains/all_domains_gold.jsonl")
    #for domain in os.listdir("../datasets/gold_result_annotations"):
     #    docs = annotations_to_spacy(f"../datasets/gold_result_annotations/{domain}/{domain}_gold.jsonl")
      #   train_dev_test_split(docs,(f"../datasets/preprocessed/{domain}/results_only"))

    #names = ["glaucoma","cardiovascular_disease","solid_tumour_cancer"]
    #docs = ["../datasets/preprocessed/out_of_domain/glaucoma_as_test/test.spacy",
     #       "../datasets/preprocessed/out_of_domain/cardiovascular_disease_as_test/test.spacy",
      #      "../datasets/preprocessed/out_of_domain/solid_tumour_cancer_as_test/test.spacy"]

    #cap_docs(docs,names, 72)
    egg = annotations_to_spacy("../datasets/expert_annotation_sets/for_preprocessing/ner/annotator1_ner_labels.jsonl")
    print("s")





