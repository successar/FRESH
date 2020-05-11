from typing import Optional, Dict, Any

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed, FeedForward
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from allennlp.training.metrics import F1Measure


@Model.register("bernoulli_bert_lstm_generator")
class SimpleGeneratorModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        max_length_ratio: float,
        requires_grad: str,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        supervise_rationale: bool = False,
    ):

        super(SimpleGeneratorModel, self).__init__(vocab, initializer, regularizer)
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
        self._supervise_rationale = supervise_rationale
        self._pos_weight = torch.Tensor([1 / max_length_ratio - 1])

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, rationale=None, **kwargs) -> Dict[str, Any]:
        # pylint: disable=arguments-differ,unused-argument

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
