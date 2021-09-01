"""
This preprocesing script parses annotated relations and entities from the prodigy jsonl output files.
Some of the code was adapted from https://github.com/explosion/projects.
"""
import json
import typer
from pathlib import Path
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
import random

msg = Printer()

MAP_LABELS = {
    "A1_RES": "A1_RES",
    "A2_RES": "A2_RES",
    "OC_RES": "OC_RES",
}


def parse_accepted(examples_path):
    loaded = open(examples_path,"r").read()
    dicts = loaded.split("\n")
    with open(examples_path, "w") as output:
        for dict in dicts:
            if dict == "\n":
                print("Parse success")
            else:
                dict = json.loads(dict)
                if dict["answer"] == "accept":
                    output.write(json.dumps(dict) + "\n")
        output.close()


def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()
    docs = []

    with json_loc.open("r", encoding="utf8") as jsonfile:
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
                    docs.append(doc)

                except KeyError as e:
                    msg.fail(f"Skipping doc because of key error")

    random.shuffle(docs) # randomise examples before train, dev, test split
    l = len(docs)
    train = docs[0:int(l*0.7)]
    dev = docs[int(l*0.7):int(l*0.85)]
    test = docs[int(l * 0.85):]

    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)
    msg.info( f"{len(train)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info( f"{len(dev)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(test_file)
    msg.info( f"{len(test)} training sentences")


if __name__ == "__main__":
    annos = "../datasets/gold/ebm_nlp/cardiovascular_disease/cardiovascular_disease_gold.jsonl"
    # typer.run(main)
    parse_accepted(annos)
