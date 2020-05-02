from allennlp.predictors.predictor import Predictor
from typing import List
from allennlp.common.util import JsonDict, sanitize

from allennlp.data import Instance

@Predictor.register("rationale_predictor")
class RationalePredictor(Predictor) :
    def _json_to_instance(self, json_dict):
        raise NotImplementedError

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        self._model.prediction_mode = True
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)