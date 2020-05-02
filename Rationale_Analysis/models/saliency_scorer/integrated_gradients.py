from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
import numpy as np

import logging
from allennlp.models.model import Model


@Model.register("integrated_gradient")
class IntegratedGradientSaliency(SaliencyScorer):
    def __init__(self, model, num_steps=10):
        self._num_steps = num_steps
        self._embedding_layer = {}

        super().__init__(model)

        self.init_from_model()

    def init_from_model(self):
        logging.info("Initialising from Model .... ")
        model = self._model['model']
        _embedding_layer = [
            x for x in list(model.modules()) if any(y in str(type(x)) for y in model.embedding_layers)
        ]
        assert len(_embedding_layer) == 1

        self._embedding_layer['embedding_layer'] = _embedding_layer[0]

    def score(self, **kwargs):
        with torch.enable_grad():
            for param in self._embedding_layer['embedding_layer'].parameters():
                param.requires_grad = True

            normal_embeddings_list = []
            gradients = 0.0
            for alpha in np.linspace(0, 1.0, num=self._num_steps, endpoint=False):
                embeddings_list = []

                def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
                    if alpha == 0.0:
                        normal_embeddings_list.append(output.detach().clone())

                    output.mul_(alpha)
                    output.retain_grad()
                    embeddings_list.append(output)

                hook = self._embedding_layer['embedding_layer'].register_forward_hook(forward_hook)
                output_dict = self._model['model'].forward(**kwargs)

                hook.remove()
                assert len(embeddings_list) == 1
                embeddings = embeddings_list[0]

                predicted_class_probs = output_dict["probs"][
                    torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"]
                ]  # (B, C)

                predicted_class_probs.sum().backward()
                gradients += embeddings.grad.detach()

            assert len(normal_embeddings_list) == 1
            embeddings = normal_embeddings_list[0]

            gradients /= self._num_steps

            gradients = (gradients * embeddings).sum(-1).detach().abs()
            gradients = gradients / gradients.sum(-1, keepdim=True)

            output_dict["attentions"] = gradients

        output_dict = self._model['model'].normalize_attentions(output_dict)

        return output_dict
