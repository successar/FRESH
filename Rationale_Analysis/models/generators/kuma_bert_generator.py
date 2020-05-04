from typing import Optional, Dict, Any

import torch
from transformers import AutoModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import F1Measure

from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel

from Rationale_Analysis.models.utils import generate_embeddings_for_pooling


@Model.register("kuma_bert_generator")
class KumaraswamyBertGenerator(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: str,
        dropout: float = 0.0,
        requires_grad: str = "none",
        pos_weight: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(KumaraswamyBertGenerator, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._bert_model = AutoModel.from_pretrained(bert_model)
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(self._bert_model.config.hidden_size, 2, bias=False)

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

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, rationale=None, **kwargs) -> Dict[str, Any]:
        #pylint: disable=arguments-differ

        bert_document = self.combine_document_query(document, query)
        
        last_hidden_states, _ = self._bert_model(
            bert_document["bert"]["wordpiece-ids"],
            attention_mask=bert_document["bert"]["wordpiece-mask"],
            position_ids=bert_document["bert"]["position-ids"],
            token_type_ids=bert_document["bert"]["type-ids"],
        )

        token_embeddings, span_mask = generate_embeddings_for_pooling(
            last_hidden_states, 
            bert_document["bert"]['document-starting-offsets'], 
            bert_document["bert"]['document-ending-offsets']
        )

        token_embeddings = util.masked_max(token_embeddings, span_mask.unsqueeze(-1) == 1, dim=2)
        token_embeddings = token_embeddings * bert_document['bert']["mask"].unsqueeze(-1)

        logits = torch.nn.functional.softplus(self._classification_layer(self._dropout(token_embeddings)))

        a, b = logits[:, :, 0], logits[:, :, 1]
        mask = bert_document['bert']['mask']

        output_dict = {}
        output_dict["a"] = a * mask
        output_dict["b"] = b * mask
        output_dict['mask'] = mask
        output_dict['wordpiece-to-token'] = bert_document['bert']['wordpiece-to-token']
        return output_dict
