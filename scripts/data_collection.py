import spacy
from glob import glob
import os
from collections import defaultdict
from Bio import Entrez
import tarfile
import json
import re
from pathlib import Path
import pickle
import operator


class ProcessEbmNlp:
    ###!!!!!! consider adding annotation adjustment functionality
    def __init__(self):
        self.queries = {}
        self.ebm_nlp = '../datasets/for_preprocessing/ebm_nlp'
        if os.path.isdir(f"{self.ebm_nlp}/ebm_nlp_2_00") == False:
            print("extracting ebm-nlp corpus -- this may take some time")
            tar = tarfile.open(f"{self.ebm_nlp}/ebm_nlp_2_00.tar.gz")
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
        print(f"Coverting raw ebm-nlp annotations into iob files")
        for fdomain in filtered_domains:
            self.generate_ebm_nlp_iob(fdomain, filtered_domains[fdomain])

            # conserve pmids which are lost when converting to iob files
            with open(f"{self.ebm_nlp}/domain_pmids/{fdomain}.pkl", 'wb') as pmid_output:
                pmid_list = [str(pmid) for pmid in filtered_domains[fdomain]]
                pickle.dump(pmid_list,pmid_output)
                pmid_output.close()

    def convert_iob_to_spacy(self):
        print(f"Coverting iob ebm-nlp annotations into spacy files")
        for filename in os.listdir(f"{self.ebm_nlp}/ebm_nlp_iob"):
            os.system(f"python -m spacy convert {self.ebm_nlp}/ebm_nlp_iob/{filename} {self.ebm_nlp}/ebm_nlp_spacy")


def add_pmids_to_doc(docs,pmids):
    new_docs = []
    for doc, pmid in zip(docs,pmids):
        doc.user_data["pmid"] = pmid
        new_docs.append(doc)
    return new_docs


def spacy_to_jsonl(spacy_data, domain_name, add_ids=None, doc_filter=None):
    """
    Converts the spacy data format used by the pipeline and its models into the jsonl format used by the prodigy
    annotation software.
    :param spacy_data: spacy files
    :add_ids: add back in pubmed id or custom id for doc
    :doc_filter: filter doc if missing certain pre-annotations
    :return: outputs jsonl files
    """
    from spacy.vocab import Vocab
    from spacy.tokens import DocBin
    from scripts.evaluate import main as evaluate
    from prodigy.components.preprocess import add_tokens

    nlp = spacy.blank("en")
    doc_bin = DocBin(store_user_data=True).from_disk(spacy_data)
    vocab = Vocab()
    docs = doc_bin.get_docs(vocab)
    if add_ids != "no":
        docs = add_pmids_to_doc(docs, add_ids)
    text = [{"text":doc.text} for doc in docs]
    tokens = [tok for tok in add_tokens(nlp, text, skip=True)]
    count = 0
    for doc in docs:
        try:
           tokens[count]["user_data"] = str(doc.user_data)
        except AttributeError:
            print("doc has no id")
            if doc_filter == "missing_ids":
                print(tokens[count].pop())
                continue
        try:
            spans = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
        except AttributeError:
            print("doc has no entities")
            if doc_filter == "missing_entities":
                print(tokens[count].pop())
                continue
        tokens[count]["spans"] = spans
        relations = []
        try:
            for key in doc._.rel:
                if doc._.rel[key]["A1_RES"] == 1.0:
                    relations.append({"child": key[1], "head": key[0], "label": "A1_RES"})
                elif doc._.rel[key]["A2_RES"] == 1.0:
                    relations.append({"child": key[1], "head": key[0], "label": "A2_RES"})
                elif doc._.rel[key]["OC_RES"] == 1.0:
                    relations.append({"child": key[1], "head": key[0], "label": "OC_RES"})
            tokens[count]["relations"] = relations
        except AttributeError:
            print("doc has no relations")
            if doc_filter == "missing_relations":
                print(tokens[count].pop())
                continue
        count += 1
    folder = f"../datasets/for_annotation/ebm_nlp/{domain_name}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(f"{folder}/unfiltered.jsonl", 'w') as domain_output:
            domain_output.write(json.dumps(tokens) + "\n")
            domain_output.close()
    return tokens


def filter_sentences(annotation_data, sent_filter, domain_name):
    """
    Filters sentences from domains
    :param annotation_data: jsonl sentences for annotation
    :param sent_filter: filter predicates,
    :param domain_name:
    :return:
    """
    from prodigy.components.preprocess import split_sentences
    import scripts.custom_functions
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("custom_sentencizer", before="parser")
    stream = split_sentences(nlp, annotation_data, min_length=30)
    folder = f"../datasets/for_annotation/ebm_nlp/{domain_name}"
    if sent_filter != {}:
        with open(f"{folder}/{sent_filter['start_parse']}.jsonl", 'w') as filter_output:
            count = 0
            section_parse = False
            for item in stream:
                if section_parse:
                    if re.match(f"{sent_filter['stop_parse']}", item["text"]):
                        section_parse = False
                    else:
                        filter_output.write(json.dumps(item) + "\n")
                        count += 1
                else:
                    if re.match(f"{sent_filter['start_parse']}", item["text"]):
                        filter_output.write(json.dumps(item) + "\n")
                        count += 1
                        section_parse = True


            print(f"{sent_filter} on {domain_name} returns {count} example sentences in {domain_name}.jsonl")
            filter_output.close()
    else:
        with open(f"{folder}/all_abstract_sentences.jsonl", 'w') as filter_output:
            for item in stream:
                filter_output.write(json.dumps(item) + "\n")
            filter_output.close()


def model_pre_annotate(data,ner_model=None,rel_model=None,filter=None):
    """
    Use existing trained models to pre-annotate data with predicted labels to speed up the annotation process
    :param ner_model: trained named entity recognition model
    :param rel_model: trained named relation extraction model
    :param data: pre-gold data for annotation
    :return: data with pre-annoation for review by user in prodigy
    """

    updated_data = []
    if ner_model != None:
        import scripts.entity_ruler
        # load ner model
        ner_nlp = spacy.load(ner_model, disable="custom_entity_ruler")
        for line in data:
            print("Adding entity pre-annotations")
            line = json.loads(line)
            ner_doc = ner_nlp(line["text"])
            line["spans"] += \
                [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in ner_doc.ents
                 if str(ent) not in filter]

            if rel_model != None:
                import scripts.custom_functions
                print("Adding relationship pre-annotations")
                # load rel model
                rel_nlp = spacy.load(rel_model)
                for name, proc in rel_nlp.pipeline:  # take rel component from pipeline
                    rel_doc = proc(ner_doc)
                    relations = []
                    try:
                        for key in rel_doc._.rel:
                            pred_rel = max(rel_doc._.rel[key].items(), key=operator.itemgetter(1)) # selects rel with highest probability
                            if pred_rel[1] > 0.1: # probability threshold for inclusion
                                relations.append({"child": key[1], "head": key[0], "label": pred_rel[0]})
                    except AttributeError:
                        print("Unable to add annotations for doc")
                    line["relations"] += relations
            updated_data.append(line)
        else:
            print("An ner model must be provided to pre-annotate")
        return updated_data


if __name__ == "__main__":
        ## default query terms used by the PICO_ner_rel system
    query_terms = {"diabetes":"diabetes",
                   "solid_tumour_cancer":"(breast cancer OR ovarian cancer OR prostate cancer OR lung cancer)",
                   "blood_cancer": "(lymphoma OR leukemia OR myeloma)",
                   "autism":"autism",
                   "cardiovascular_disease":
                       "(stroke OR myocardial infarction OR thrombosis OR heart attack OR heart failure)"}
    #pp_ebm_nlp = PreprocessEbmNlp()
    #for term in query_terms:
    #     pp_ebm_nlp.add_queries(query_terms[term], term)
    #pp_ebm_nlp.output_iob()
    #pp_ebm_nlp.convert_iob_to_spacy()
    prepro_ebm_dir = "../datasets/for_preprocessing/ebm_nlp"
    anno_ebm_dir = "../datasets/for_annotation/ebm_nlp"
    spacy_dir = f"{prepro_ebm_dir}/ebm_nlp_spacy"
    pmid_dir = f"{prepro_ebm_dir}/domain_pmids"
    fanno_dir = "../datasets/for_annotation/ebm_nlp"
    ner_model = "../trained_model/ner/glaucoma/model-best"
    rel_model = "../trained_model/rel_pipeline/glaucoma/model-best"
    filter = ["POPU","INTV","OC"]

    for sfn, pfn in zip(os.listdir(spacy_dir), os.listdir(pmid_dir)):

     #   fn_split = re.search("(?<=(nlp_))[_a-z]*(?=\.)", sfn)
      #  name = fn_split.group(0)

       # pmids = pickle.load(open(f'{pmid_dir}/{pfn}', 'rb'))
        #spacy_files = f"{spacy_dir}/{sfn}"

        #if os.path.isdir(f"../datasets/for_annotation/ebm_nlp/{name}") == False:
         #   print("Converting spacy files to jsonl for prodigy annotation")
          #  spacy_to_jsonl(spacy_files, name, add_ids=pmids, doc_filter="missing_entities")

     for domain in os.listdir(anno_ebm_dir):
         annotation_data = json.load(open(f"../datasets/for_annotation/ebm_nlp/{domain}/unfiltered.jsonl", "r"))
         result_sent_filter = {"start_parse": "RESULT", "stop_parse": "CONCLUSION"}
         no_filter = {}
         filter_sentences(annotation_data, no_filter, domain)

        #for domain in os.listdir(fanno_dir):
     #   data = open(f"{fanno_dir}/{domain}/RESULT.jsonl","r")
      #  preannotated = model_pre_annotate(data, ner_model=ner_model, rel_model=rel_model, filter=filter)
       # with open(f"{fanno_dir}/{domain}/RESULT_preannotated.jsonl", 'w') as preannotated_output:
        #        for line in preannotated:
         #               preannotated_output.write(json.dumps(line) + "\n")
          #      preannotated_output.close()




