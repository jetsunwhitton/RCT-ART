from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from scripts.evaluate import main as evaluate
from prodigy.components.preprocess import add_tokens
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
import spacy
from spacy.tokens import DocBin
import json
from glob import glob
import os
from collections import defaultdict
import re
from Bio import Entrez
import pickle
import tarfile

def extract_ents_rels(data):
    doc_bin = DocBin(store_user_data=True).from_disk(data)
    vocab = Vocab()
    text = [{"text":doc.text} for doc in doc_bin.get_docs(vocab)]
    tokens = [tok for tok in add_tokens(nlp, text, skip=True)]
    count = 0
    for doc in doc_bin.get_docs(vocab):
        spans = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
        tokens[count]["spans"] = spans
        relations = []
        for key in doc._.rel:
            if doc._.rel[key]["A1_MEASURE"] == 1.0:
                relations.append({"child": key[1], "head": key[0], "label": "A1"})
            elif doc._.rel[key]["A2_MEASURE"] == 1.0:
                relations.append({"child": key[1], "head": key[0], "label": "A2"})
            else:
                pass
        tokens[count]["relations"] = relations
        count += 1
    return tokens


class PreprocessEbmNlp:
    ###!!!!!! consider adding annotation adjustment functionality
    def __init__(self):
        self.queries = {}
        self.ebm_nlp = '../datasets/for_preprocessing/ebm_nlp'
        if os.path.isdir(f"{self.ebm_nlp}/ebm_nlp_2_00") == False:
            print("extracting ebm-nlp corpus -- this may take some time")
            tar = tarfile.open(f"{self.ebm_nlp}/ebm_nlp _2_00.tar.gz")
            tar.extractall(path=self.ebm_nlp)
        self.token_fnames = glob(f'{self.ebm_nlp}/ebm_nlp_2_00/documents/*.tokens')

    def add_queries(self, query, domain):
        """
        Add query for filtering the ebm-nlp corpus into a domain
        :param query: string for querying ids on pubmed
        :param domain: single-word name for query and resulting domain, should be same as query if one query
        is one word
        """
        self.queries[domain] = query

    def remove_query(self, domain):
        self.queries.pop(domain)

    def clear_queries(self):
        self.queries.clear()

    def ebm_nlp_pmid_batch(self):
        """
        Batches the ebm-nlp corpus pmids for joining with queries
        :return: list of batched pmid strings
        """
        pmids = "../datasets/for_preprocessing/ebm_nlp/ebm-nlp_pmids.txt"
        try:
            ebm_nlp_pmids = open(pmids, "r")
        except FileNotFoundError:
            self.get_ebm_nlp_pmids(pmids)
            ebm_nlp_pmids = open(pmids, "r")
        pmids = ebm_nlp_pmids.read().split(" ")
        l = len(pmids) # takes length of pmids for indexing
        batches = []
        seperator = "," #  seperator for join function below needed to convert pmid batch to string
        for i in range(10,0,-1):
            batches.append(seperator.join(pmids[int(l*(1/i-0.1)):int(l*1/i)])) #  converts to string and adds to batch list
        return batches

    def pmid_db_query(self, batch, query):
        """
        Query pubmed database for pmids
        :param query: query for pubmed database
        :return: pmids matching query
        """
        Entrez.email = 'jetsun.whitton.20@ucl.ac.uk'
        post = Entrez.epost("pubmed", id=batch)
        p = Entrez.read(post) # query keys of posted pmids
        search = Entrez.esearch("pubmed",query,usehistory='y',WebEnv=p['WebEnv'],query_key=p['QueryKey'],retmax=500)
        results = Entrez.read(search)
        return results

    def fname_to_pmid(self, fname):
        """helper function to get pmids from filenames"""
        pmid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]
        return pmid

    def get_ebm_nlp_pmids(self, output):
        """output pmids of all ebm_nlp_"""
        pmid_output = open(output, 'w')
        for fname in self.token_fnames:
            pmid_output.write(self.token_fnames(fname) + " ")
        pmid_output.close()

    def generate_ebm_nlp_iob(self, domain_name, domain):
        """
        Formats the filtered ebm-nlp domains into iob files.
        Domain functionality was added to iob formatter code from original Nye et al. github:
        https://github.com/bepnye/EBM-NLP
        :param domain_name: the name of the filtered domain
        :param domain: the domain set of ebm-nlp data i.e. the labelled abstracts
        :return: iob files for each filtered domain
        """
        PIO = ['participants', 'interventions', 'outcomes']  # default annotations
        PHASES = ['starting_spans'] # default annotations
        id_to_tokens = {}
        id_to_pos = {}
        for fname in self.token_fnames:
          pmid = self.fname_to_pmid(fname)
          if pmid in domain:
              try:
                  tokens = open(fname).read().split()
                  tags   = open(fname.replace('tokens', 'pos')).read().split()
                  id_to_tokens[pmid] = tokens
                  id_to_pos[pmid] = tags
              except UnicodeDecodeError:
                  print(f"UnicodeDecodeError with: {fname}")
        batch_to_labels = {}
        for phase in PHASES:
          batch_to_labels[phase] = {}
          for pio in PIO:
            batch_to_labels[phase][pio] = {}
            print(f'Reading files for {phase} {pio}')
            for fdir in ['train', 'test/gold']:
              ann_fnames = glob(f'{self.ebm_nlp}/ebm_nlp_2_00/annotations/aggregated/{phase}/{pio}/{fdir}/*.ann')
              for fname in ann_fnames:
                  pmid = self.fname_to_pmid(fname)
                  if pmid in domain:
                    batch_to_labels[phase][pio][pmid] = open(fname).read().split()

        batch_groups = [(['starting_spans'], ['participants', 'interventions', 'outcomes'])]
        for phases, pio in batch_groups:
          id_to_labels_list = defaultdict(list)
          batch_to_ids = {}
          for phase in phases:
            for e in pio:
                batch_to_ids = batch_to_labels[phase][e].keys()

                for pmid, labels in batch_to_labels[phase][e].items():
                  labels = [f'{l}_{e[0]}' for l in labels]
                  id_to_labels_list[pmid].append(labels)

          fout = open(f'{self.ebm_nlp}/ebm_nlp_iob/ebm_nlp_{domain_name}.iob', 'w')
          for pmid in batch_to_ids:
            try:
              fout.write('-DOCSTART- -X- O O\n\n')
              begin = True
              tokens = id_to_tokens[pmid]
              poss = id_to_pos[pmid]
              per_token_labels = zip(*id_to_labels_list[pmid])
              prev = ""
              for i, (token, pos, labels) in enumerate(zip(tokens, poss, per_token_labels)):
                final_label = 'O'
                for l in labels:
                  if l[0] != '0' and (begin == True or l[2] != prev):
                      if l[2] == 'p': final_label = "B-POPU"
                      if l[2] == 'i': final_label = "B-INTV"
                      if l[2] == 'o': final_label = "B-OC"
                      begin = False
                      prev = l[2]
                      break
                  elif l[0] != '0':
                      if l[2] == 'p': final_label = "I-POPU"
                      if l[2] == 'i': final_label = "I-INTV"
                      if l[2] == 'o': final_label = "I-OC"
                      prev = l[2]
                      break
                  elif l[2] == 'o':
                      begin = True
                  else:
                      pass
                if token == '.':
                    fout.write('%s %s %s\n' % (token, pos, "O"))
                    fout.write('\n')
                    begin = True
                else:
                    fout.write('%s %s %s\n' % (token, pos, final_label))
            except KeyError:
                print(pmid, " error")

    def output_iob(self):
        print("Batching PMIDs for querying")
        batches = self.ebm_nlp_pmid_batch()
        filtered_domains = defaultdict(list)
        for domain in self.queries:
            count = 0
            for batch in batches:
                print(f"Making query: {self.queries[domain]} | PMID batch: {count}")
                count += 1
                handle = self.pmid_db_query(batch, self.queries[domain])
                filtered_domains[domain] += handle["IdList"]
        for fdomain in filtered_domains:
            self.generate_ebm_nlp_iob(fdomain, filtered_domains[fdomain])


if __name__ == "__main__":
    ## default query terms used by the PICO_ner_rel system
    query_terms = {"diabetes":"diabetes",
                   "solid_tumour_cancer":"(breast cancer OR ovarian cancer OR prostate cancer OR lung cancer)",
                   "blood_cancer": "(lymphoma OR leukemia OR myeloma)",
                   "autism":"autism",
                   "cardiovascular_disease":
                       "(stroke OR myocardial infarction OR thrombosis OR heart attack OR heart failure)"}
    pp_ebm_nlp = PreprocessEbmNlp()
    for term in query_terms:
        pp_ebm_nlp.add_queries(query_terms[term], term)
    pp_ebm_nlp.output_iob()

    nlp = spacy.blank("en")

    # Assets
    test_data = "../assets/test.spacy"
    train_data = "../assets/train.spacy"
    dev_data = "../assets/dev.spacy"

    #get_ebm_nlp_pmids("ebm-nlp_pmids.txt")
    #with open("../assets/results_sentences.jsonl", 'w') as output:
     #   for item in extract_ents_rels(test_data):
      #      output.write(json.dumps(item) + "\n")
       # for item in extract_ents_rels(train_data):
        #    output.write(json.dumps(item) + "\n")
        #for item in extract_ents_rels(dev_data):
         #   output.write(json.dumps(item) + "\n")
       # output.close()


