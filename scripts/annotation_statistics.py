import numpy as np
import spacy, os
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.vocab import Vocab
import sys

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def get_all_token_ner_labels(annotator_path):
    nlp = spacy.load("../trained_models/biobert/ner/all_domains/model-best")
    doc_bin = DocBin(store_user_data=True).from_disk(annotator_path)
    #print(doc_bin.__len__())
    docs = doc_bin.get_docs(nlp.vocab)
    all_token_labels = []
    for gold in docs:
        all_token_labels += [ner.ent_type_ for ner in gold]
    #print(len(all_token_labels))
    return all_token_labels


def get_all_pair_rel_labels(annotator_path):
    rel = spacy.load("../trained_models/biobert/rel/all_domains/model-best")
    doc_bin = DocBin(store_user_data=True).from_disk(annotator_path)
    docs = doc_bin.get_docs(rel.vocab)
    all_ent_pair_rel_labels = []
    value_list = []
    for gold in docs:
        for value, rel_dict in gold._.rel.items():
            value_list.append(value)
            label = [k for (k, v) in gold._.rel[value].items() if v == 1.0]
            if label != []:
                all_ent_pair_rel_labels.append(label[0])
            else:
                all_ent_pair_rel_labels.append('')
    print(value_list)
    return all_ent_pair_rel_labels


def create_ner_annotation_matrix(label_list):
    # initialise matrix with first rows
    if label_list[0] == "INTV":
        am = np.matrix([[1, 0, 0, 0]])
    elif label_list[0] == "OC":
        am = np.matrix([[0, 1, 0, 0]])
    elif label_list[0] == "MEAS":
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


def create_rel_annotation_matrix(label_list):
    # initialise matrix with first rows
    if label_list[0] == "A1_RES":
        am = np.matrix([[1, 0, 0, 0]])
    elif label_list[0] == "A2_RES":
        am = np.matrix([[0, 1, 0, 0]])
    elif label_list[0] == "OC_RES":
        am = np.matrix([[0, 0, 1, 0]])
    else:
        am = np.matrix([[0, 0, 0, 1]])

    # add more rows
    for label in label_list[1:]:
        if label == "A1_RES":
            am = np.vstack([am, [1, 0, 0, 0]])
        elif label == "A2_RES":
            am = np.vstack([am, [0, 1, 0, 0]])
        elif label == "OC_RES":
            am = np.vstack([am, [0, 0, 1, 0]])
        else:
            am = np.vstack([am, [0, 0, 0, 1]])
    return am


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


if __name__ == "__main__":

    # REL LABEL IAA

    # annotator data input paths
    d_rel = "../datasets/2_expert_annotation_sets/for_statistics/rel"
    rel_anno_paths = os.listdir(d_rel)

    # construct matrices out of rel annotations
    rel_anno_matrix_list = []
    for path in rel_anno_paths:
        all_rel_labels = get_all_pair_rel_labels(os.path.join(d_rel,path))
        rel_anno_matrix_list.append(create_rel_annotation_matrix(all_rel_labels))

    # sum all annotator matrices and convert back to array for statistics
    all_rel_annotator_matrix = np.squeeze(np.asarray(sum(rel_anno_matrix_list)))

    # remove rows where all made no label
    all_rel_annotator_matrix = np.delete(all_rel_annotator_matrix, np.where((all_rel_annotator_matrix[:, 3] == 3))[0], axis=0)

    print(all_rel_annotator_matrix, all_rel_annotator_matrix.shape)

    # output fleiss_kappa score
    print(fleiss_kappa(all_rel_annotator_matrix))


    # ENTITY LABEL IAA

    # annotator data input paths
    d_ner = "../datasets/2_expert_annotation_sets/for_statistics/ner"
    ner_anno_paths =  os.listdir(d_ner)

    ner_anno_matrix_list = []
    for path in ner_anno_paths:
        all_token_labels = get_all_token_ner_labels(os.path.join(d_ner,path))
        ner_anno_matrix_list.append(create_ner_annotation_matrix(all_token_labels))

    # sum all annotator matrices and convert back to array for statistics
    all_ner_annotator_matrix = np.squeeze(np.asarray(sum(ner_anno_matrix_list)))

    # remove rows where all made no label
    all_ner_annotator_matrix = np.delete(all_ner_annotator_matrix, np.where((all_ner_annotator_matrix[:, 3] == 3))[0], axis=0)

    print(all_ner_annotator_matrix, all_ner_annotator_matrix.shape)

    print(fleiss_kappa(all_ner_annotator_matrix))



