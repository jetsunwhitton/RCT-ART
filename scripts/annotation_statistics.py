import numpy as np
import spacy, operator, csv, os
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.vocab import Vocab
import sys

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

def create_token_lookup(path):
    nlp = spacy.load("../trained_models/biobert/ner/all_domains/model-best")
    doc_bin = DocBin(store_user_data=True).from_disk(path)
    docs = doc_bin.get_docs(nlp.vocab)
    lookup = {}
    count = 0
    for gold in docs:
        sent_pos = 0
        for tok in gold:
            lookup[str(count)] = (tok, sent_pos, gold.text)
            count += 1
            sent_pos += 1
    # print(len(all_token_labels))
    return lookup


def create_annotation_matrix(label_list):
    # initialise matrix with first rows
    #print(label_list)
    if label_list[1] == "INTV":
        am = np.matrix([[1, 0, 0, 0]])
    elif label_list[1] == "OC":
        am = np.matrix([[0, 1, 0, 0]])
    elif label_list[1] == "MEAS":
        am = np.matrix([[0, 0, 1, 0]])
    else:
        am = np.matrix([[0, 0, 0, 1]])

    # add more rows
    for label in label_list[1:]:
        if label == "INTV":
            am = np.vstack([am, [1, 0, 0, 0]])
        elif label == "OC":
            am = np.vstack([am, [0, 1, 0, 0]])
        elif label == "MEAS":
            am = np.vstack([am, [0, 0, 1, 0]])
        else:
            am = np.vstack([am, [0, 0, 0, 1]])
    return am


def identify_disagreement(matrix, lookup):
    count = 0
    disagree_list = []
    for row in matrix:
        if 3 not in row:
            disagree_list.append((row,lookup[str(count)]))
        count += 1
    return disagree_list

def fleiss_kappa(table, method='fleiss'):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    n_sub, n_cat = table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    # assume fully ranked
    assert n_total == n_sub * n_rat

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    if method == 'fleiss':
        p_mean_exp = (p_cat * p_cat).sum()
    elif method.startswith('rand') or method.startswith('unif'):
        p_mean_exp = 1 / n_cat

    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return kappa


# annotator data input paths
a1_input = "../datasets/expert_annotation_sets/for_statistics/ner/annotator1_ner_labels.spacy"
a2_input = "../datasets/expert_annotation_sets/for_statistics/ner/annotator2_ner_labels.spacy"
a3_input = "../datasets/expert_annotation_sets/for_statistics/ner/annotator3_ner_labels.spacy"
anno_paths = [a1_input, a2_input,a3_input]

def get_all_token_labels(annotator_path):
    nlp = spacy.load("../trained_models/biobert/ner/all_domains/model-best")
    doc_bin = DocBin(store_user_data=True).from_disk(annotator_path)
    #print(doc_bin.__len__())
    docs = doc_bin.get_docs(nlp.vocab)
    all_token_labels = []
    for gold in docs:
        all_token_labels += [ent.ent_type_ for ent in gold]
    #print(len(all_token_labels))
    return all_token_labels

anno_matrix_list = []
for path in anno_paths:
    all_token_labels= get_all_token_labels(path)
    anno_matrix_list.append(create_annotation_matrix(all_token_labels))

# create token lookup dictionary
lookup = create_token_lookup(a1_input)

#print(lookup)

# sum all annotator matrices and convert back to array for statistics
all_annotator_matrix = np.squeeze(np.asarray(sum(anno_matrix_list)))

# remove all entries where all made no label
all_annotator_matrix = np.delete(all_annotator_matrix, np.where((all_annotator_matrix[:, 3] == 3))[0], axis=0)

# intv matrix
#all_annotator_matrix = np.delete(all_annotator_matrix, np.where((all_annotator_matrix[:, 0] == 0))[0], axis=0)

# oc matrix
#all_annotator_matrix = np.delete(all_annotator_matrix, np.where((all_annotator_matrix[:, 1] == 0))[0], axis=0)

# meas matrix
#all_annotator_matrix = np.delete(all_annotator_matrix, np.where((all_annotator_matrix[:, 2] == 0))[0], axis=0)

# print(all_annotator_matrix, all_annotator_matrix.shape)

print(fleiss_kappa(all_annotator_matrix))

disagree_list = identify_disagreement(all_annotator_matrix,lookup)

output = open("disagree_list.txt", "w")


for out in disagree_list:
    try:
        output.write(str(out) + "\n")
    except:
        output.write(str("error") + "\n")

output.close()
print("success")
