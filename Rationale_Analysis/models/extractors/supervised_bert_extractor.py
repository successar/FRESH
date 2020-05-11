from typing import Optional, Dict, Any

import torch
from transformers import AutoModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import F1Measure, Average

from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from Rationale_Analysis.models.utils import generate_embeddings_for_pooling

from Rationale_Analysis.models.thresholders.top_k import TopKThresholder


@Model.register("supervised_bert_extractor")
class SupervisedBertExtractor(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: str,
        dropout: float = 0.0,
        requires_grad: str = "none",
        max_length_ratio: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(SupervisedBertExtractor, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._bert_model = AutoModel.from_pretrained(bert_model)
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(self._bert_model.config.hidden_size, 1, bias=False)

        if requires_grad in ["none", "all"]:
            for param in self._bert_model.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in self._bert_model.named_parameters():
                found = any([regex in name for regex in model_name_regexes])
                param.requires_grad = found

        for n, v in self._bert_model.named_parameters():
            if n.startswith("classifier"):
                v.requires_grad = True

        self._token_prf = F1Measure(1)
        self._rationale_length = Average()

        self._pos_weight = torch.Tensor([1 / max_length_ratio - 1])
        self._extractor = TopKThresholder(max_length_ratio=max_length_ratio)

        initializer(self)

    def forward(
        self, document, query=None, label=None, metadata=None, rationale=None, **kwargs
    ) -> Dict[str, Any]:
        # pylint: disable=arguments-differ

        bert_document = self.combine_document_query(document, query)

        last_hidden_states, _ = self._bert_model(
            bert_document["bert"]["wordpiece-ids"],
            attention_mask=bert_document["bert"]["wordpiece-mask"],
            position_ids=bert_document["bert"]["position-ids"],
            token_type_ids=bert_document["bert"]["type-ids"],
        )

        token_embeddings, span_mask = generate_embeddings_for_pooling(
            last_hidden_states,
            bert_document["bert"]["document-starting-offsets"],
            bert_document["bert"]["document-ending-offsets"],
        )

        token_embeddings = util.masked_max(token_embeddings, span_mask.unsqueeze(-1) == 1, dim=2)
        token_embeddings = token_embeddings * bert_document["bert"]["mask"].unsqueeze(-1)

        logits = self._classification_layer(self._dropout(token_embeddings))

        probs = torch.sigmoid(logits)[:, :, 0]
        mask = bert_document["bert"]["mask"]

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
