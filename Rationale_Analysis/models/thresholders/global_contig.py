from Rationale_Analysis.models.thresholders.base_thresholder import Thresholder
from Rationale_Analysis.models.thresholders.global_objective import max_contig
from allennlp.models.model import Model

import numpy as np

@Model.register("global_contig")
class GlobalTopKRationaleExtractor(Thresholder) :
    def __init__(self, max_length_ratio: float, min_inst_ratio: float=-1.0):
        self._max_length_ratio = max_length_ratio
        self._min_inst_ratio = min_inst_ratio if min_inst_ratio >= 0.0 else (max_length_ratio / 2.0)
        super().__init__()

    def forward(self, attentions, metadata) :
        rationales = self.extract_rationale(attentions=attentions, metadata=metadata)
        output_dict = {'metadata' : metadata, 'rationale' : rationales}
        return output_dict
 
    def extract_rationale(self, attentions, metadata):
        attentions = attentions.cpu().data.numpy()
        sentences = [x["tokens"] for x in metadata]
        rationales = []
        top_spans = max_contig(attentions, [len(s) for s in sentences], self._max_length_ratio, self._min_inst_ratio)
        for s,(i, j) in enumerate(top_spans):
            sentence = [x.text for x in sentences[s]]
                        
            rationales.append({
                'document' : " ".join(sentence[i:j]),
                'spans' : [{'span' : (i, j), 'value' : np.sum(attentions[s][i:j])}],
                'metadata' : None
            })

        return rationales