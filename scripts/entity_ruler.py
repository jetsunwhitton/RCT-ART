"""
An
"""
from spacy import load
import re
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.scorer import Scorer
from spacy.language import Language
from spacy.pipeline import EntityRuler


@Language.factory("custom_entity_ruler")
def create_custom_entity_ruler(nlp: Language, name: str):
    return custom_entity_ruler(nlp)

class custom_entity_ruler:

    def __init__(self, nlp):
        ####  set pattern variables to add to entity_ruler  ####

        # procedure patterns
        with open("./datasets/procedure_suffix_regex.txt") as regex_input:
            PROC_SFX = [{"LOWER": {"REGEX": regex_input.read()}}, {"TEXT": "-", "OP": "?"}]
        G_DENOM = [{"LOWER": "group"},
                   {"TEXT": {"REGEX": "(([Aa]|[Bb]|([Oo]ne)|1|([Tt]wo)|2)(?!.))"}}]  # group with denomination
        A_DENOM = [{"LOWER": "arm"},
                   {"TEXT": {"REGEX": "(([Aa]|[Bb]|([Oo]ne)|1|([Tt]wo)|2)(?!.))"}}]  # arm with denomination
        ADJ_GROUP = [{"POS": "ADJ", "OP": "+"}, {"LOWER": "group"}]  # adj + group
        NOUN_GROUP = [{"POS": "NOUN", "OP": "+"}, {"LOWER": "group"}]  # noun + group
        ADJ_ARM = [{"POS": "ADJ", "OP": "+"}, {"LOWER": "arm"}]  # adj + arm
        NOUN_ARM = [{"POS": "NOUN", "OP": "+"}, {"LOWER": "arm"}]  # noun + arm
        COMBO = [{"LOWER": {"REGEX": "plus|\+|with"}}]
        G_OR_A = [{"LOWER": {"REGEX": "((group|arm)(?!s))"}, "OP": "?"}]

        # number patterns
        NUM = [{"LIKE_NUM": True}]
        NUM_BRACKET = [{"TEXT": "(", "OP": "?"}, {"TEXT": "[", "OP": "?"}, {"LIKE_NUM": True}, {"TEXT": ")", "OP": "?"},
                       {"TEXT": "]", "OP": "?"}]
        NUM_OP = [{"LIKE_NUM": True, "OP": "?"}]
        NONE = [{"LOWER": "none"}]
        NO = [{"LOWER": "no"}]

        # percentage patterns
        PERC = [{"LIKE_NUM": True}, {"IS_SPACE": True, "OP": "?"}, {"TEXT": "%"}]  # percentage alone
        PERC_ABS = [{"OP": "?"}, {"IS_PUNCT": True, "OP": "?"}] + PERC + [
            {"IS_PUNCT": True, "OP": "?"}]  # percentage after absolute value

        # interval patterns
        IVAL_SEP = [{"LIKE_NUM": True}, {"TEXT": "+"}, {"TEXT": "/"}, {"TEXT": "-"},
                    {"LIkE_NUM": True}]  # seperate plus minus signs
        IVAL_COMB = [{"LIKE_NUM": True}, {"TEXT": "±"}, {"LIkE_NUM": True}]  # combined plus minus signs
        ONE_TOKEN_IVAL = [{"TEXT": {"REGEX": "(\d+\.?\d*±\d+\.?\d*)|(\d+\.?\d*\+\/\-\d+\.?\d*)"}}]

        # range patterns
        RANGE = [{"TEXT": {"REGEX": "(to)|[-]"}}]

        # unit patterns
        UNIT = [{"LOWER": "mm", "OP": "?"},
                {"TEXT": {"REGEX": "(mm)?[Hh][Gg]|mg\/m[Ll]|mg|m[Ll]"}, "OP": "?"}]

        # non-result patterns to ignore
        TIME = NUM_OP + \
               [{"TEXT": {"REGEX": "([Yy]ears?)|([Mm]onths?)|([Ww]eeks?)|([Dd]ays?)|([Hh]ours?)|([Mm]inutes?)"}}] + \
               NUM_OP + [{"LOWER": "and", "OP": "?"}] + NUM_OP  # time point
        TIME_RANGE = NUM + RANGE + TIME  # time range
        PVAL = [{"LOWER": "p"}, {"TEXT": {"REGEX": "<|="}}] + NUM  # pvalue
        COMP_STAT = [{"TEXT": {"REGEX": "HR|OR"}}, {"IS_PUNCT": True, "OP": "?"},
                     {"IS_SPACE": True, "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}] + NUM  # comparative statistics

        #  pattern variables for entity_ruler in different combinations to create more complex rule-based matching
        patterns = [{"label": "INTV", "pattern": PROC_SFX + COMBO + PROC_SFX + G_OR_A},  # named combination
                    {"label": "INTV", "pattern": PROC_SFX + PERC + G_OR_A},
                    # named treatment with concentration
                    {"label": "INTV", "pattern": PROC_SFX + G_OR_A},  # named treatment
                    {"label": "INTV", "pattern": G_DENOM},  # generic group with denomination e.g. a/b/1/2
                    {"label": "INTV", "pattern": A_DENOM},  # generic arm with denomination e.g. a/b/1/2
                    {"label": "INTV", "pattern": ADJ_GROUP},  # adjective phrase group
                    {"label": "INTV", "pattern": NOUN_GROUP},  # noun phrase group
                    {"label": "INTV", "pattern": ADJ_ARM},  # adjective phrase arm
                    {"label": "INTV", "pattern": NOUN_ARM},  # noun phrase arm
                    {"label": "COMSTAT", "pattern": COMP_STAT},  # ignore pattern
                    {"label": "PVAL", "pattern": PVAL},  # ignore pattern
                    {"label": "TIMR", "pattern": TIME_RANGE},  # ignore pattern
                    {"label": "TIME", "pattern": TIME},  # ignore pattern
                    {"label": "MEAS", "pattern":
                        ONE_TOKEN_IVAL + UNIT + RANGE + ONE_TOKEN_IVAL + UNIT},  # range result
                    {"label": "MEAS", "pattern": IVAL_SEP + UNIT + RANGE + IVAL_SEP + UNIT},  # range result
                    {"label": "MEAS", "pattern":
                        NUM + UNIT + RANGE + NUM + UNIT},  # range result
                    {"label": "MEAS", "pattern": ONE_TOKEN_IVAL + UNIT},  # interval result
                    {"label": "MEAS", "pattern": IVAL_SEP + UNIT},  # interval result
                    {"label": "MEAS", "pattern": IVAL_COMB + UNIT},  # interval result
                    {"label": "MEAS", "pattern": NUM + NUM_OP + UNIT + PERC_ABS},  # perc result
                    {"label": "MEAS", "pattern": PERC + NUM_OP},  # perc result
                    {"label": "MEAS", "pattern": PERC + NUM_BRACKET},  # perc result
                    {"label": "MEAS", "pattern": PERC},  # perc result
                    {"label": "MEAS", "pattern": NUM + NUM_OP + UNIT},  # absolute (with optional interval) result
                    {"label": "MEAS", "pattern": NONE},  # absolute number result
                    {"label": "MEAS", "pattern": NO}]  # absolute number result
        # Add patterns
        self.ruler = EntityRuler(nlp, overwrite_ents=False)
        self.ruler.add_patterns(patterns)
        self.ruler.initialize(lambda: [], nlp=nlp, patterns=patterns)

    def __call__(self, doc: Doc) -> Doc:
        # Remove ignore patterns from doc when called -- these patterns are detected as useful for context
        ents = [ent for ent in self.ruler(doc).ents if ent.label_ not in ["COMSTAT","PVAL","TIMR","TIME"]]
        doc.ents = ents
        return doc

# load pre-trained model for feature extraction such as pos tagging
#nlp = load("C:\\Users\\jetsu\\1. Degree stuff\\COMP0073 Summer project\\spacy_re\\training\\model-best")

#nlp.add_pipe("custom_entity_ruler")

# add domain specific special cases to tokeniser
#bslash_minus = [{"ORTH": "/"}, {"ORTH": "-"}]

#nlp.tokenizer.add_special_case("/-", bslash_minus)

#def ent_match(doc,patterns):
    #ents = []
    #for pattern in patterns:
        #print(doc.text)
        #print([toks for toks in doc])
        #for match in re.finditer(pattern["regex"], doc.text):
        #    start, end = match.span()
       #     print(f"The matched text: '{doc.text[start:end]}'")
      #      span = doc.char_span(start, end, label=pattern["label"], alignment_mode="expand")
     #       if span != None:
    #            ents.append(span)
   # print([(ent, ent.label_) for ent in ents])
  #  doc.ents = ents
 #   return [(ent, ent.label_) for ent in doc.ents]


#with open("C:\\Users\\jetsu\\1. Degree stuff\\COMP0073 Summer project\\spacy_re\pre-processing\\assets\\clean_sentences_2.txt") as input:
 #  for line in input:
  #     doc = nlp(line)
   #    print([(ent, ent.label_) for ent in doc.ents])
    #   print([(toks.text, toks.pos_)  for toks in doc])
     #  print(doc)


#def create_examples(nlp, test_data):
    #doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    #docs = doc_bin.get_docs(nlp.vocab)
    examples = []
    #for gold in docs:
     #   examples.append(Example(nlp(gold.text, disable="ner"), gold))
    #for example in examples:
      #  print("\n\n",example.text)
     #   print("predicted-->",[(ent.text,ent.label_) for ent in example.predicted.ents])
    #    print("gold-->", [(ent.text, ent.label_) for ent in example.reference.ents])
   # scorer = Scorer(nlp)
  #  scores = scorer.score_spans(examples, 'ents')
 #   print(scores)

#test_data = "../../data/test.spacy"

#create_examples(nlp, test_data)

#text = ("""IOPs at the 3 time points assessed during 15% the 12-week visit ranged from 17.4 to 18.6 mm Hg for PF tafluprost and 17.9 to 18.5 mm Hg for PF timolol""")

#@Language.component("result_entity_basic_elements")
#def measure_elements(doc, nlp):
 #   ruler = EntityRuler(nlp)

    #ruler.add_patterns(patterns)
    #doc = ruler(doc)
    #return doc

#print(ent_match(doc,patterns))

#def rule_matcher(input):
 #   for text in input:
  #      doc = nlp(text)
   #     doc = ruler_one(doc)
    #    print([(ent, ent.label_) for ent in doc.ents])
     #   print(doc)

# Construction from class
#ruler = EntityRuler(nlp, overwrite_ents=True)

#match = Matcher(nlp.vocab)

#num_rule.add_patterns([{"label": "result", "pattern": [{"LIKE_NUM": True}{"LIKE_NUM},{"TEXT":"to"},{"LIKE_NUM": True}]}])
#doc = nlp(text)
#print ([(ent, ent.label_) for ent in doc.ents])

