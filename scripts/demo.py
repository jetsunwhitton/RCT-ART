import spacy_streamlit, spacy, operator
import streamlit as st
from spacy import displacy
from spacy.pipeline import merge_entities
# make the factory work
from rel_pipe import make_relation_extractor, score_relations
# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
from tabulate import relation_extraction, tabulate_pico_entities
import base64

# set page config
st.set_page_config(
	page_title="RCT-ART",
	page_icon="logo.jpg"
)

st.sidebar.image("logo.jpg")
st.sidebar.markdown("RCT-ART is an NLP pipeline built with spaCy for converting clinical trial result sentences into tables through jointly extracting intervention, outcome and outcome measure entities and their relations. ")
st.sidebar.subheader("Current constraints:")
st.sidebar.markdown("""
                    - Only abstracts from studies with 2 trial arms
                    - Must be a sentence with study results
                    - Sentence must contain at least least one intervention (e.g. drug name), outcome description (e.g. blood pressure) and non-comparative outcome measure)
                    """)
st.title("Demo")
st.header("Randomised Controlled Trial Abstract Result Tabulator")

ner_model = "trained_models/biobert/ner/all_domains/model-best"
rel_model = "trained_models/biobert/rel/all_domains/model-best"

default_text = "Somnolence , the most frequently reported adverse event , was noted in 72.5 % versus 7.7 % of subjects ( risperidone vs placebo ) and seemed manageable with dose/dose-schedule modification ."

st.subheader("Enter result sentence for analysis")
text = st.text_area("Input should follow constraints outlined in sidebar", default_text, height=200)
nlp = spacy.load("trained_models/biobert/ner/all_domains/model-best")
ent_doc = nlp(text)

st.subheader("NER analysis")
spacy_streamlit.visualize_ner(
    ent_doc,
    labels=["INTV", "OC", "MEAS"],
    show_table=False,
    title=False
)

rel_doc = relation_extraction(rel_model,[ent_doc])[0]

deps = {"words": [],"arcs": []}

for tok in rel_doc:
    deps["words"].append({"text": tok.text, "tag": tok.ent_type_})

for key in rel_doc._.rel:
    rel = rel_doc._.rel[key]  # get relation
    pred_rel = max(rel.items(), key=operator.itemgetter(1))  # selects relation type with highest probability
    if pred_rel[1] > 0.5:  # includes relation if above set threshold for probability
        if key[0] > key[1] and rel_doc[key[1]].ent_type_ != "MEAS":
            deps["arcs"].append({"start": key[1], "end": key[0], "label":  pred_rel[0], "dir": "right"})
        elif key[0] > key[1]:
            deps["arcs"].append({"start": key[1], "end": key[0], "label": pred_rel[0], "dir": "left"})
        elif rel_doc[key[1]].ent_type_ != "MEAS":
            deps["arcs"].append({"start": key[0], "end": key[1], "label": pred_rel[0], "dir": "left"})
        else:
            deps["arcs"].append({"start": key[0], "end": key[1], "label": pred_rel[0], "dir": "right"})

html = displacy.render(deps, style="dep", manual=True, options={'distance':80})

st.subheader("RE analysis")
st.write(spacy_streamlit.util.get_svg(html), unsafe_allow_html=True)

heading_properties = [('font-size', '16px')]

cell_properties = [('font-size', '16px')]

dfstyle = [dict(selector="th", props=heading_properties),dict(selector="td", props=cell_properties)]

df = tabulate_pico_entities(rel_doc)
print(rel_doc._.rel)

#df.style.set_table_styles([cell_hover, index_names, headers])

st.subheader("Tabulation")
st.table(df.style.set_table_styles(dfstyle))

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="result_sentence.csv">Download csv file</a>'
    return href

st.markdown(get_table_download_link(df), unsafe_allow_html=True)

