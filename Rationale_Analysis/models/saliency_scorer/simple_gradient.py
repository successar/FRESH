from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
import logging
from allennlp.models.model import Model

@Model.register("simple_gradient")
class GradientSaliency(SaliencyScorer) :  
    def __init__(self, model) :
        self._embedding_layer = {}
        super().__init__(model)

        self.init_from_model()

    def init_from_model(self) :
        logging.info("Initialising from Model .... ")
        model = self._model['model']

        _embedding_layer = [x for x in model.modules() if any(x == y for y in model.embedding_layers)]
        assert len(_embedding_layer) == 1

        self._embedding_layer['embedding_layer'] = _embedding_layer[0]


    def score(self, **kwargs) :
        with torch.enable_grad() :
            self._model['model'].train()
            
            if hasattr(self._model['model'], 'prepare_for_gradient') :
                self._model['model'].prepare_for_gradient()
            else :
                for param in self._embedding_layer['embedding_layer'].parameters():
                    param.requires_grad = True

            embeddings_list = []
            def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
                embeddings_list.append(output)
                output.retain_grad()

            hook = self._embedding_layer['embedding_layer'].register_forward_hook(forward_hook)
            output_dict = self._model['model'].forward(**kwargs)

            hook.remove()
            assert len(embeddings_list) == 1
            embeddings = embeddings_list[0]

            predicted_class_probs = output_dict["probs"][
                torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"].detach()
            ]  # (B, C)


            predicted_class_probs.sum().backward(retain_graph=True)

            gradients = ((embeddings * embeddings.grad).sum(-1).detach()).abs()
            gradients = gradients / gradients.sum(-1, keepdim=True)

            output_dict['attentions'] = gradients

        output_dict = self._model['model'].normalize_attentions(output_dict)

        return output_dict