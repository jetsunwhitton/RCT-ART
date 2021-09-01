import spacy

from spacy.tokens import DocBin, Doc
from spacy.training.example import Example

from rel_component.scripts.evaluate import main as evaluate

from rel_component.scripts.rel_model import create_relation_model, create_classification_layer, create_instances

from pandas import DataFrame


#text = ("""Both dose strengths of bimatoprost implant were noninferior to timolol in IOP lowering after each administration.

#Mean diurnal IOP was 24.0, 24.2, and 23.9 mmHg at baseline and from 16.5-17.2, 16.5-17.0, and 17.1-17.5 mmHg through week 12 in the 10-μg implant, 15-μg implant, and timolol groups, respectively.

#The incidence of corneal and inflammatory TEAEs of interest (e.g., corneal endothelial cell loss, iritis) was higher with bimatoprost implant than timolol and highest with the 15-μg dose strength.

#Incidence of corneal TEAEs increased after repeated treatment; with 3 administrations at fixed 16-week intervals, incidence of ≥20% CECD loss was 10.2% (10-μg implant) and 21.8% (15-μg implant).


#Mean best-corrected visual acuity (BCVA) was stable; 3 implant-treated subjects with corneal TEAEs had >2-line BCVA loss at their last visit.""")

#text = "Enrolled into the 2 studies were 1167 patients. Treatment with netarsudil q.d. produced clinically and statistically significant reductions from baseline intraocular pressure (P < .001), and was noninferior to timolol in the per-protocol population with maximum baseline IOP < 25 mm Hg in both studies (ROCKET-2, primary outcome measure and population, ROCKET-1, post hoc outcome measure). Netarsudil b.i.d. was also noninferior to timolol (ROCKET-2). The most frequent adverse event was conjunctival hyperemia, the incidence of which ranged from 50% (126/251, ROCKET-2) to 53% (108/203, ROCKET-1) for netarsudil q.d., 59% (149/253, ROCKET-2) for netarsudil b.i.d., and 8% (17/208, ROCKET-1) to 11% (27/251, ROCKET-2) for timolol (P < .0001 for netarsudil vs timolol)."

#nlp0 = spacy.load("en_core_web_sm", disable=["ner"])
#nlp0.add_pipe("senten")
#doc = nlp0(text)

nlp = spacy.load("C:\\Users\\jetsu\\1. Degree stuff\\COMP0073 Summer project\\spacy_re\\training\\model-best")
test_data = "data/test.spacy"

def print_doc(nlp, data):
    doc_bin = DocBin(store_user_data=True).from_disk(data)
    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        print(doc)
        print(doc._.rel)

print_doc(nlp, test_data)

def ner_processing(nlp, data):
    doc_bin = DocBin(store_user_data=True).from_disk(data)
    docs = doc_bin.get_docs(nlp.vocab)
    ent_processed_docs = []
    for doc in docs:
        doc.ents = nlp(doc.text).ents
        ent_processed_docs.append(doc)
    return ent_processed_docs

#docbin = DocBin(docs=ner_processing(nlp, test_data), store_user_data=True)
#docbin.to_disk("data/test_ner_processed.spacy")


#nlp2 = spacy.load("C:\\Users\\jetsu\\1. Degree stuff\\COMP0073 Summer project\\spacy_re\\rel_component\\training\\model-best")
#evaluate(nlp2,ent_processed)

#doc_bin = DocBin(store_user_data=True).from_disk(Path)

#for doc in doc_bin.get_docs(ent_detector.vocab):
 #   for ents in doc.ents:
  #      print(ents, ents.label_)

#print(doc_bin)


Path = "data/test_ner_processed.spacy"


def rel_processing(nlp, test_data: Path):
    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in doc],
            spaces=[t.whitespace_ for t in doc],
        )
        pred.ents = doc.ents
        for name, proc in nlp.pipeline:
            pred = proc(pred)
        rel_predictions = ([(value, rel_dict) for value, rel_dict in pred._.rel.items()
                            if rel_dict["A1_RES"] > 0.1 or rel_dict["A2_RES"] > 0.1 or rel_dict["OC_RES"] > 0.1])
        rel_tab_dict = {"Outcome":{},"Arm 1": {},"Arm 2": {}}
        for entry in rel_predictions:
            if entry[1]['A1_RES'] > entry[1]['A2_RES']:
                intervention = list(filter(lambda x: x.start == entry[0][0], pred.ents))[0]
                result = list(filter(lambda x: x.start == entry[0][1], pred.ents))[0]
                rel_tab_dict["Arm 1"]["Intervention"] = intervention
                if "Results" in rel_tab_dict["Arm 1"]:
                    rel_tab_dict["Arm 1"]["Results"].append(result)
                else:
                    rel_tab_dict["Arm 1"]["Results"] = [result]
            elif entry[1]['A2_MEASURE'] > entry[1]['A1_MEASURE']:
                intervention = list(filter(lambda x: x.start == entry[0][0], pred.ents))[0]
                result = list(filter(lambda x: x.start == entry[0][1], pred.ents))[0]
                rel_tab_dict["Arm 2"] = {"Intervention": intervention}
                if "Results" in rel_tab_dict["Arm 2"]:
                    rel_tab_dict["Arm 2"]["Results"].append(result)
                else:
                    rel_tab_dict["Arm 2"]["Results"] = [result]
        print("\n\n", doc.text)
        #print(rel_tab_dict)
        print(DataFrame.from_dict(rel_tab_dict, orient='columns', dtype=None, columns=None))

#rel_processing(nlp2, Path)


def main(nlp, test_data: Path, print_details: bool):
    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    examples = []
    for gold in docs:
        for x in gold.ents:
            print (x.label_)
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        pred.ents = gold.ents
        for name, proc in nlp.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, gold))
        print(type(pred))
        # Print the gold and prediction, if gold label is not 0
        if print_details:
            print()
            print(f"Text: {gold.text}")
            print(f"spans: {[(e.start, e.text, e.label_) for e in pred.ents]}")
            for value, rel_dict in pred._.rel.items():
                gold_labels = [k for (k, v) in gold._.rel[value].items() if v == 1.0]
                if gold_labels:
                    print(
                        f" pair: {value} --> gold labels: {gold_labels} --> predicted values: {rel_dict}"
                    )
            print()



#for name, proc in nlp.pipeline:

 #   pred = proc(doc)
  #  for rel_dict in pred._.rel.items():
   #     if rel_dict[1]['A1_MEASURE'] > 0.05 or rel_dict[1]['A2_MEASURE']>0.05:
    #        print(f"spans: {[(e.start, e.text, e.label_) for e in pred.ents]}")
     #       print(rel_dict)


#main(nlp, Path, True)