from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin, Doc
from spacy.language import Language

# make the factory work
from scripts.rel_pipe import make_relation_extractor

# make the config work
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
from scripts.entity_ruler import custom_entity_ruler


@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):
        # Define sentence start if . + not numeric (e.g decimal point) token after
        if token.text == "." and not doc[i + 1].text.isnumeric():
            doc[i + 1].is_sent_start = True
        else:
            # Explicitly set sentence start to False otherwise, to tell
            # the parser to leave those tokens alone
            doc[i + 1].is_sent_start = False
    return doc


# This function was sourced from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
@spacy.registry.readers("Gold_ents_Corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


# This function was adapated from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    """Custom reader that keeps the tokenization of the gold data,
    and also adds the gold PICO annotations as we do not attempt to predict these."""
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)
    for gold in docs:
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        pred.ents = gold.ents
        yield Example(pred, gold)
