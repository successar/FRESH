from typing import Optional, Dict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.training.metrics import FBetaMeasure, CategoricalAccuracy


class RationaleBaseModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(RationaleBaseModel, self).__init__(vocab, regularizer)
        self._vocabulary = vocab
        self._f1_metric = FBetaMeasure()
        self._accuracy = CategoricalAccuracy()

        self.prediction_mode = False

        initializer(self)

    def forward(self, document, query=None, labels=None, metadata=None, **kwargs):
        # pylint: disable=arguments-differ

        raise NotImplementedError

    def make_output_human_readable(self, output_dict):
        output_dict = self._decode(output_dict)
        output_labels = self._vocabulary.get_index_to_token_vocabulary("labels")

        predicted_labels, gold_labels = [], []
        for p, g in zip(output_dict["predicted_label"], output_dict["label"]):
            predicted_labels.append(output_labels[int(p)])
            gold_labels.append(output_labels[int(g)])

        output_dict["predicted_label"] = predicted_labels
        output_dict["label"] = gold_labels
        output_dict["annotation_id"] = [d['annotation_id'] for d in output_dict['metadata']]

        del output_dict['metadata']

        return output_dict

    def _call_metrics(self, output_dict):
        self._f1_metric(output_dict["probs"], output_dict["gold_labels"])
        self._accuracy(output_dict["probs"], output_dict["gold_labels"])

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._f1_metric.get_metric(reset)
        macro_avg = {'macro_' + k: sum(v) / len(v) for k, v in metrics.items()}
        output_labels = self._vocabulary.get_index_to_token_vocabulary("labels")
        output_labels = [output_labels[i] for i in range(len(output_labels))]

        class_metrics = {}
        for k, v in metrics.items():
            assert len(v) == len(output_labels)
            class_nums = dict(zip(output_labels, v))
            class_metrics.update({k + "_" + str(kc): x for kc, x in class_nums.items()})

        class_metrics.update({"accuracy": self._accuracy.get_metric(reset)})
        class_metrics.update(macro_avg)
        modified_class_metrics = {}

        for k, v in class_metrics.items():
            if k in ["accuracy", "macro_fscore"]:
                modified_class_metrics[k] = v
            else:
                modified_class_metrics["_" + k] = v

        modified_class_metrics["validation_metric"] = class_metrics["macro_fscore"]

        return modified_class_metrics

    def normalize_attentions(self, output_dict):
        """
        In case, attention is over subtokens rather than at token level. 
        Combine subtoken attention into token attention.
        """

        return output_dict

    def combine_document_query(self, document, query):
        reader = document[0]["reader_object"]
        device = next(self.parameters()).device
        return {
            k: ({x: y.to(device) for x, y in v.items()} if type(v) == dict else v.to(device))
            for k, v in reader.combine_document_query(document, query, self._vocabulary).items()
        }

    # Because Allennlp loads models with strict=True 
    # but encoder_generator type models requires 
    # rationale extractor without keepsake params
    # Reader need not worry.
    def load_state_dict(self, state_dict, strict=True) :
        super().load_state_dict(state_dict, strict=False)

