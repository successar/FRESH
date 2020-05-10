from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_Analysis.models.classifiers.base_model import RationaleBaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention.additive_attention import AdditiveAttention

@Model.register("bert_lstm_classifier")
class EncoderAttentionClassifier(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        requires_grad: str,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(EncoderAttentionClassifier, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")
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

        seq2seq_size = self._seq2seq_encoder.get_output_dim()
        self._attention = AdditiveAttention(vector_dim=seq2seq_size, matrix_dim=seq2seq_size)

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

        self._vector = torch.nn.Parameter(torch.randn((1, self._seq2seq_encoder.get_output_dim(),)))

        self.embedding_layers = [self._text_field_embedder]

        initializer(self)

    def forward(self, document, query=None, label=None, metadata=None, **kwargs) -> Dict[str, Any]:
        # pylint: disable=arguments-differ,unused-argument

        tensorized_document = self.combine_document_query(document, query)
        
        embedded_text = self._text_field_embedder(tensorized_document)
        mask = util.get_text_field_mask(tensorized_document)

        embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
        attentions = self._attention(vector=self._vector, matrix=embedded_text, matrix_mask=mask)

        embedded_text = embedded_text * attentions.unsqueeze(-1) * mask.unsqueeze(-1)
        embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

        logits = self._classification_layer(embedded_vec)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}

        output_dict['logits'] = logits
        output_dict["probs"] = probs
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["attentions"] = attentions
        output_dict["metadata"] = metadata

        if label is not None :
            loss = F.cross_entropy(logits, label)
            output_dict["loss"] = loss
            self._call_metrics(output_dict)
            
        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}
        new_output_dict["predicted_label"] = output_dict["predicted_labels"].cpu().data.numpy()
        new_output_dict["label"] = output_dict["gold_labels"].cpu().data.numpy()
        new_output_dict["metadata"] = output_dict["metadata"]
        return new_output_dict

    def prepare_for_gradient(self) :
        for name, param in self._text_field_embedder.named_parameters():
            if '11' in name :
                param.requires_grad = True