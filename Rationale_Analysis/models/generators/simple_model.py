from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.time_distributed import TimeDistributed

@Model.register("simple_generator_model")
class SimpleGeneratorModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(SimpleGeneratorModel, self).__init__(vocab, regularizer)
        self._vocabulary = vocab
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = feedforward_encoder.get_output_dim()

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 2)

        initializer(self)

    def forward(self, document) -> Dict[str, Any]:
        embedded_text = self._text_field_embedder(document)
        mask = util.get_text_field_mask(document).float()

        embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
        embedded_text = self._dropout(self._feedforward_encoder(embedded_text))

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.Softmax(dim=-1)(logits)[:, :, 1]

        output_dict = {}

        output_dict["probs"] = probs * mask

        predicted_rationale = (probs > 0.5).long()
        output_dict['predicted_rationale'] = predicted_rationale * mask
        output_dict["prob_z"] = probs * mask

        return output_dict

    