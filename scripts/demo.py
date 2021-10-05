import spacy_streamlit, spacy, operator
import streamlit as st
from spacy import displacy
from spacy.pipeline import merge_entities
# make the factory work
from rel_pipe import make_relation_extractor, score_relations
# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
from tabulate import relation_extraction, tabulate_pico_entities

# set page config
st.set_page_config(
	page_title="RCT-ART",
	page_icon="logo.jpg"
)


ner_model = "trained_models/biobert/ner/all_domains/model-best"
rel_model = "trained_models/biobert/rel/all_domains/model-best"

default_text = "Somnolence , the most frequently reported adverse event , was noted in 72.5 % versus 7.7 % of subjects ( risperidone vs placebo ) and seemed manageable with dose/dose-schedule modification ."


text = st.text_area("Text to analyze", default_text, height=200)
nlp = spacy.load("trained_models/biobert/ner/all_domains/model-best")
ent_doc = nlp(text)
spacy_streamlit.visualize_ner(
    ent_doc,
    labels=["INTV", "OC", "MEAS"],
    show_table=False,
    title="ICO entities",
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

st.write(spacy_streamlit.util.get_svg(html), unsafe_allow_html=True)


cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#ffffb3')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #000066; color: white;'
}

df = tabulate_pico_entities(rel_doc)

df.style.set_table_styles([cell_hover, index_names, headers])

st.dataframe(df.style)
