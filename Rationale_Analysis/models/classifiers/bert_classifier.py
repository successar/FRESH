from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel
from Rationale_Analysis.models.utils import generate_embeddings_for_pooling


@Model.register("bert_classifier")
class BertClassifier(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: str,
        dropout: float = 0.0,
        requires_grad: str = "none",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(BertClassifier, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")
        self._bert_config = AutoConfig.from_pretrained(bert_model, output_attentions=True)
        self._bert_model = AutoModel.from_pretrained(bert_model, config=self._bert_config)

        self._dropout = nn.Dropout(dropout)
        self._classifier = nn.Linear(self._bert_model.config.hidden_size, self._num_labels)

        self.embedding_layers = [self._bert_model.embeddings]

        if requires_grad in ["none", "all"]:
            for param in self._bert_model.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in self._bert_model.named_parameters():
                found = any([regex in name for regex in model_name_regexes])
                param.requires_grad = found

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, **kwargs) -> Dict[str, Any]:
        # pylint: disable=arguments-differ,unused-argument

        bert_document = self.combine_document_query(document, query)       

        _, pooled_output, attentions = self._bert_model(
            bert_document["bert"]["wordpiece-ids"],
            attention_mask=bert_document["bert"]["wordpiece-mask"],
            position_ids=bert_document["bert"]["position-ids"],
            token_type_ids=bert_document["bert"]["type-ids"],
        )

        logits = self._classifier(self._dropout(pooled_output))

        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}
        attentions = attentions[-1][:, :, 0, :].mean(1)

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["attentions"] = attentions
        output_dict["metadata"] = metadata

        output_dict["document-starting-offsets"] = bert_document["bert"]["document-starting-offsets"]
        output_dict["document-ending-offsets"] = bert_document["bert"]["document-ending-offsets"]

        if label is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, label)
            output_dict["loss"] = loss
            self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        return new_output_dict

    def normalize_attentions(self, output_dict) -> None:
        attentions = output_dict['attentions'].unsqueeze(-1)
        document_token_starts = output_dict['document-starting-offsets']
        document_token_ends = output_dict['document-ending-offsets']
        
        token_attentions, token_mask = generate_embeddings_for_pooling(attentions, document_token_starts, document_token_ends)

        token_attentions = (token_attentions * token_mask.unsqueeze(-1)).squeeze(-1).sum(-1)
        output_dict["attentions"] = token_attentions / token_attentions.sum(-1, keepdim=True)

        return output_dict
