from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
from allennlp.models.model import Model

@Model.register("wrapper")
class WrapperSaliency(SaliencyScorer) :    
    def score(self, **kwargs) :
        output_dict = self._model['model'].forward(**kwargs)
        output_dict = self._model['model'].normalize_attentions(output_dict)
        assert 'attentions' in output_dict, "No key 'attentions' in output_dict"
        return output_dict