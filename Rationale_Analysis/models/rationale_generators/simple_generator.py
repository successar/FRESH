from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder

from allennlp.training.metrics import F1Measure


@Model.register("simple_middle_model")
class BertMiddleModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        use_crf: bool = False,
        pos_weight: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(BertMiddleModel, self).__init__(vocab, regularizer)
        self._vocabulary = vocab
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = feedforward_encoder.get_output_dim()

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 2)

        self._use_crf = use_crf

        self._pos_weight = torch.Tensor([1 / (1 - pos_weight), 1 / pos_weight])
        self._pos_weight = torch.nn.Parameter(self._pos_weight / self._pos_weight.min())
        self._pos_weight.requires_grad = False

        if use_crf:
            self._crf = ConditionalRandomField(num_tags=2)

        self._token_prf = F1Measure(1)

        initializer(self)

    def forward(self, document, query=None, rationale=None, metadata=None, label=None) -> Dict[str, Any]:
        embedded_text = self._text_field_embedder(document)
        mask = util.get_text_field_mask(document).float()

        embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
        embedded_text = self._dropout(self._feedforward_encoder(embedded_text))

        logits = self._classification_layer(embedded_text)

        if self._use_crf:
            best_paths = self._crf.viterbi_tags(logits, mask=document["mask"])
            best_paths = [b[0] for b in best_paths]
            best_paths = [x + [0] * (logits.shape[1] - len(x)) for x in best_paths]
            best_paths = torch.Tensor(best_paths).to(logits.device) * document["mask"]
        else:
            best_paths = (logits[:, :, 1] > 0.5).long() * document["mask"]

        output_dict = {}

        output_dict["predicted_rationales"] = best_paths
        output_dict["mask"] = document["mask"]
        output_dict["metadata"] = metadata

        if rationale is not None:
            if self._use_crf:
                output_dict["loss"] = -self._crf(logits, rationale, document["mask"])
            else:
                output_dict["loss"] = (
                    (
                        F.cross_entropy(
                            logits.view(-1, logits.shape[-1]),
                            rationale.view(-1),
                            reduction="none",
                            weight=self._pos_weight,
                        )
                        * document["mask"].view(-1)
                    )
                    .sum(-1)
                    .mean()
                )

            best_paths = best_paths.unsqueeze(-1)
            best_paths = torch.cat([1 - best_paths, best_paths], dim=-1)
            self._token_prf(best_paths, rationale, document["mask"])
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
        return dict(zip(["p", "r", "f1"], metrics))

    def decode(self, output_dict) :
        rationales = self.extract_rationale(output_dict)
        new_output_dict = {}

        new_output_dict['rationale'] = rationales
        new_output_dict['document'] = [r['document'] for r in rationales]
        
        if 'query' in output_dict['metadata'][0] :
            output_dict['query'] = [m['query'] for m in output_dict['metadata']]

        for m in output_dict["metadata"]:
            if 'convert_tokens_to_instance' in m :
                del m["convert_tokens_to_instance"]

        new_output_dict['label'] = [m['label'] for m in output_dict['metadata']]
        new_output_dict['metadata'] = output_dict['metadata']

        return new_output_dict

