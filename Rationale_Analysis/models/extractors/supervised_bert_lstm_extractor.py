from typing import Optional, Dict, Any

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward, TimeDistributed

from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from allennlp.training.metrics import F1Measure, Average
from Rationale_Analysis.models.thresholders.top_k import TopKThresholder


@Model.register("supervised_bert_lstm_extractor")
class SupervisedBertLstmExtractor(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        requires_grad: str,
        dropout: float = 0.0,
        max_length_ratio: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(SupervisedBertLstmExtractor, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._text_field_embedder = text_field_embedder

        if requires_grad in ["none", "all"]:
            for param in self._text_field_embedder.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in self._text_field_embedder.named_parameters():
                found = any([regex in name for regex in model_name_regexes])
                param.requires_grad = found

        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._feedforward_encoder = TimeDistributed(feedforward_encoder)
        self._classifier_input_dim = feedforward_encoder.get_output_dim()

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 1, bias=False)

        self._token_prf = F1Measure(1)
        self._rationale_length = Average()

        self._pos_weight = torch.Tensor([1 / max_length_ratio - 1])
        self._extractor = TopKThresholder(max_length_ratio=max_length_ratio)

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, rationale=None, **kwargs) -> Dict[str, Any]:
        bert_document = self.combine_document_query(document, query)

        embedded_text = self._text_field_embedder(bert_document)
        mask = util.get_text_field_mask(bert_document)

        embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
        embedded_text = self._dropout(self._feedforward_encoder(embedded_text))

        logits = self._classification_layer(embedded_text)
        
        probs = torch.sigmoid(logits)[:, :, 0]
        mask = mask.float()

        output_dict = {}
        output_dict["probs"] = probs * mask
        output_dict["mask"] = mask

        output_dict["metadata"] = metadata
        output_dict["document"] = document

        if rationale is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                rationale,
                reduction="none",
                pos_weight=self._pos_weight.to(rationale.device),
            )
            loss = ((loss * mask).sum(-1) / mask.sum(-1)).mean()
            output_dict["loss"] = loss

            self._token_prf(
                torch.cat([1 - probs.unsqueeze(-1), probs.unsqueeze(-1)], dim=-1),
                rationale.long(),
                mask == 1,
            )

            predicted_rationale = (probs > 0.5).long() * mask
            self._rationale_length(((predicted_rationale * mask).sum(-1).float() / mask.sum(-1)).mean())

        return output_dict

    def extract_rationale(self, output_dict):
        rationales = []
        sentences = [x["tokens"] for x in output_dict["metadata"]]
        predicted_rationales = output_dict["predicted_rationales"].cpu().data.numpy()
        for path, words in zip(predicted_rationales, sentences):
            path = list(path)[: len(words)]
            words = [x.text for x in words]
            starts, ends = [], []
            path.append(0)
            for i in range(len(words)):
                if path[i - 1 : i] == [0, 1]:
                    starts.append(i)
                if path[i - 1 : i] == [1, 0]:
                    ends.append(i)

            assert len(starts) == len(ends)
            spans = list(zip(starts, ends))

            rationales.append(
                {
                    "document": " ".join([w for i, w in zip(path, words) if i == 1]),
                    "spans": [{"span": (s, e), "value": 1} for s, e in spans],
                    "metadata": None,
                }
            )

        return rationales

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._token_prf.get_metric(reset)
        metrics = dict(zip(["p", "r", "f1"], metrics))
        metrics.update({"rlength": float(self._rationale_length.get_metric(reset))})
        return metrics

    def make_output_human_readable(self, output_dict):
        rationales = self._extractor.extract_rationale(attentions=output_dict['probs'], document=output_dict['document'], as_one_hot=False)
        new_output_dict = {}

        new_output_dict["predicted_rationale"] = rationales
        new_output_dict["document"] = [r["document"] for r in rationales]

        if "query" in output_dict["metadata"][0]:
            new_output_dict["query"] = [m["query"] for m in output_dict["metadata"]]

        new_output_dict["label"] = [m["label"] for m in output_dict["metadata"]]
        new_output_dict["annotation_id"] = [m["annotation_id"] for m in output_dict["metadata"]]

        return new_output_dict
