"""
This script defines the relation extractor component for use as part of a spaCy pipeline.
It includes functions for training the relation extraction model, as well as for evaluating
its performance both alone and jointly after named entity recognition
"""
from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from spacy.scorer import PRFScore
from thinc.types import Floats2d
import numpy, operator
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer


Doc.set_extension("rel", default={}, force=True)
msg = Printer()

# This object was sourced from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)

# This object was sourced from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
def make_relation_extractor(
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)

# This class was sourced from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
class RelationExtractor(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError("Only strings can be added as labels to the RelationExtractor")
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        # check that there are actually any candidate instances in this batch of examples
        total_instances = len(self.model.attrs["get_instances"](doc))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        get_instances = self.model.attrs["get_instances"]
        total_instances = sum([len(get_instances(doc)) for doc in docs])
        if total_instances == 0:
            msg.info("Could not determine any instances in any docs - can not make any predictions.")
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        get_instances = self.model.attrs["get_instances"]
        for doc in docs:
            for (e1, e2) in get_instances(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
                c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # check that there are actually any candidate instances in this batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(self.model.attrs["get_instances"](eg.predicted))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc.")
            return losses

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths = self._examples_to_truth(examples)
        gradient = scores - truths
        mean_square_error = (gradient ** 2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                relations = example.reference._.rel
                for indices, label_dict in relations.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError("Call begin_training with relevant entities and relations "
                             "annotated in at least a few reference examples!")
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances == 0:
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_instances"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        return score_relations(examples, self.threshold)

# This function was adapted from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
# it now provide scores for joint ner and relation extraction
# as well as for relation classes
def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score ner and rel in a batch of examples."""
    micro_prf = PRFScore()
    a1_res_prf = PRFScore()
    a2_res_prf = PRFScore()
    oc_res_prf = PRFScore()
    for example in examples:
        gold_rels = example.reference._.rel
        pred_rels = example.predicted._.rel
        gold_ents = [e.text for e in example.reference.ents]
        pred_ents = [e.text for e in example.predicted.ents]
        assessed_ents = []
        for key, pred_dict in pred_rels.items():
            # checks if entity pair in gold list for scoring relations
            if key in gold_rels.keys():
                gold_labels = [k for (k, v) in gold_rels[key].items() if v == 1.0]
                for k, v in pred_dict.items():
                    if v >= threshold:
                        if k in gold_labels:
                            micro_prf.tp += 1
                            if k == "A1_RES": a1_res_prf.tp += 1
                            elif k == "A2_RES": a2_res_prf.tp += 1
                            else: oc_res_prf.tp += 1
                        else:
                            micro_prf.fp += 1
                            if k == "A1_RES": a1_res_prf.fp += 1
                            elif k == "A2_RES": a2_res_prf.fp += 1
                            else: oc_res_prf.fp += 1
                    else:
                        if k in gold_labels:
                            micro_prf.fn += 1
                            if k == "A1_RES": a1_res_prf.fn += 1
                            elif k == "A2_RES": a2_res_prf.fn += 1
                            else: oc_res_prf.fn += 1

            # keys match the entity indexes of the gold annos, if an entity pair
            # is not in the gold list this second part of the code evaluates
            # if there are correct entities with no relations mapped
            else:
                pred_rel = max(pred_dict.items(), key=operator.itemgetter(1))
                if pred_rel[1] > 0.5:
                    micro_prf.fp += 1 # relation mapped for incorrect entity pair
                else:
                    parent_ent = list(filter(lambda x: x.start == key[0], example.predicted.ents))[0].text
                    child_ent = list(filter(lambda x: x.start == key[1], example.predicted.ents))[0].text
                    if parent_ent not in assessed_ents:
                        if parent_ent in gold_ents:
                            micro_prf.tp += 1 # correctly labelled entity and no relation
                        else:
                            micro_prf.fp += 1  # incorrectly labelled entity
                        assessed_ents.append(parent_ent)
                    if child_ent not in assessed_ents:
                        if child_ent in gold_ents:
                            micro_prf.tp += 1  # correctly labelled entity and no relation
                        else:
                            micro_prf.fp += 1 # incorrectly labelled entity
                        assessed_ents.append(child_ent)

        # counts entity false negatives by checking gold ents not appearing in pred ents
        leftover_ents = [ents_left for ents_left in gold_ents if ents_left not in pred_ents]
        for missed_ent in leftover_ents:
                micro_prf.fn += 1 # gold entity missed

    return {
        "rel_micro_p": micro_prf.precision,
        "rel_micro_r": micro_prf.recall,
        "rel_micro_f": micro_prf.fscore,
        "a1_res_p": a1_res_prf.precision,
        "a1_res_r":a1_res_prf.recall,
        "a1_res_f": a1_res_prf.fscore,
        "a2_res_p": a2_res_prf.precision,
        "a2_res_r": a2_res_prf.recall,
        "a2_res_f": a2_res_prf.fscore,
        "oc_res_p": oc_res_prf.precision,
        "oc_res_r": oc_res_prf.recall,
        "oc_res_f": oc_res_prf.fscore
    }
