from typing import Optional, Dict, Any

import torch
from transformers import AutoModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import F1Measure

from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from Rationale_Analysis.models.utils import generate_embeddings_for_pooling


@Model.register("bernoulli_bert_generator")
class BernoulliBertGenerator(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: str,
        max_length_ratio: float,
        dropout: float = 0.0,
        requires_grad: str = "none",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        supervise_rationale: bool = False,
    ):

        super(BernoulliBertGenerator, self).__init__(vocab, initializer, regularizer)
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
        self._supervise_rationale = supervise_rationale
        self._pos_weight = torch.Tensor([1 / max_length_ratio - 1])

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
        predicted_rationale = (probs > 0.5).long()

        output_dict["predicted_rationale"] = predicted_rationale * mask
        output_dict["prob_z"] = probs * mask

        if rationale is not None and self._supervise_rationale:
            rat_mask = rationale.sum(1) > 0
            if rat_mask.sum().long() == 0:
                output_dict["loss"] = 0.0
            else:
                rat_mask = rat_mask.bool()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[rat_mask].squeeze(-1),
                    rationale[rat_mask],
                    reduction="none",
                    pos_weight=self._pos_weight.to(rationale.device),
                )
                loss = ((loss * mask[rat_mask]).sum(-1) / mask[rat_mask].sum(-1)).mean()
                output_dict["loss"] = loss
                self._token_prf(
                    torch.cat([1 - probs[rat_mask].unsqueeze(-1), probs[rat_mask].unsqueeze(-1)], dim=-1),
                    rationale[rat_mask].long(),
                    mask[rat_mask] == 1,
                )

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        try:
            metrics = self._token_prf.get_metric(reset)
        except:
            metrics = {"_rp": 0, "_rr": 0, "_rf1": 0}
            return metrics
        return dict(zip(["_rp", "_rr", "_rf1"], metrics))
