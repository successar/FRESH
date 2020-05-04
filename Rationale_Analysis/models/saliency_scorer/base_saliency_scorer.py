from allennlp.models.model import Model
import torch

class SaliencyScorer(Model) :
    def __init__(self, model) :
        self._model = { 'model' : model } #This is so the model is protected from Saliency_Scorer's state_dict !
        for v in self._model['model'].parameters() :
            v.requires_grad = False

        self._model['model'].prediction_mode = True
        self._model['model'].eval()

        super().__init__(self._model['model'].vocab)
        self._keepsake_param = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, **inputs) :
        output_dict = self.score(**inputs)
        return output_dict

    def make_output_human_readable(self, output_dict) :
        assert "attentions" in output_dict
        assert "metadata" in output_dict

        new_output_dict = {k:[] for k in output_dict['metadata'][0].keys()}
        for example in output_dict['metadata'] :
            for k, v in example.items() :
                new_output_dict[k].append(v)

        tokens = [example.split() for example in new_output_dict['document']]

        attentions = output_dict['attentions'].cpu().data.numpy()

        assert len(tokens) == len(attentions)
        assert max([len(s) for s in tokens]) == attentions.shape[-1]

        new_output_dict['saliency'] = [[round(float(x), 5) for x in list(m)[:len(tok)]] for m, tok in zip(attentions, tokens)]
            
        return new_output_dict
        
    def score(self, **inputs) :
        raise NotImplementedError

    def init_from_model(self) :
        pass