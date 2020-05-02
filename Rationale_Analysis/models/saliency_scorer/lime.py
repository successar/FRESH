from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
import numpy as np
import logging
from allennlp.models.model import Model
from lime.lime_text import LimeTextExplainer
from allennlp.data.dataset import Batch
from allennlp.data.tokenizers import Token

from math import ceil


@Model.register("lime")
class LimeSaliency(SaliencyScorer):
    def __init__(self, model, desired_length: float, batch_size: int):
        super().__init__(model)
        self._desired_length = float(desired_length)
        self._batch_size = batch_size
        self.init_from_model()

        output_labels = self._model["model"]._vocabulary.get_token_to_index_vocabulary("labels").keys()
        self._explainer = LimeTextExplainer(class_names=output_labels, split_expression=" ", bow=False)

        self._model["model"].eval()

    def score(self, metadata, **kwargs):
        assert "convert_tokens_to_instance" in metadata[0], breakpoint()
        tokens = metadata[0]["tokens"]
        keep_tokens = metadata[0]["keep_tokens"]
        selection_tokens = [t.text for i, t in zip(keep_tokens, tokens) if i == 0]
        query_tokens = [t for i, t in zip(keep_tokens[1:], tokens[1:]) if i == 1]

        output_dict = self._model["model"].forward(metadata=metadata, **kwargs)

        num_features = ceil(self._desired_length * len(selection_tokens))
        del kwargs["document"]
        del kwargs["label"]

        predicted_label = output_dict["predicted_labels"][0].item()

        def predict_proba(text_list):
            tokens_and_query = [
                [tokens[0]] + [Token(t) for t in selected_tokens.split(" ") if t != "UNKWORDZ"] + query_tokens
                for selected_tokens in text_list
            ]

            probs = []
            for i in range(0, len(tokens_and_query), self._batch_size):
                document = self.regenerate_tokens(
                    tokens_and_query[i : i + self._batch_size], metadata, device=self._keepsake_param.device
                )
                output = self._model["model"].forward(document=document, label=None, **kwargs)
                probs.append(output["probs"].cpu().data.numpy())
            return np.concatenate(probs, axis=0)

        explanation = self._explainer.explain_instance(
            " ".join(selection_tokens),
            predict_proba,
            num_features=num_features,
            labels=(predicted_label,),
            num_samples=500,
        )

        weights = explanation.as_map()[predicted_label]
        saliency = [0 for _ in range(len(selection_tokens))]
        for f, w in weights:
            saliency[f] = 1

        saliency = torch.Tensor([[0] + saliency + [0] * len(query_tokens)]).to(self._keepsake_param.device)

        output_dict["attentions"] = saliency

        return output_dict

    def regenerate_tokens(self, tokens_list, metadata, device):
        instances = []
        for words in tokens_list:
            instance = metadata[0]["convert_tokens_to_instance"](words)
            instances.append(instance)

        batch = Batch(instances)
        batch.index_instances(self._model["model"]._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(device) for k, v in batch["document"].items()}
