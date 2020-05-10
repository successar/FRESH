from Rationale_Analysis.models.thresholders.base_thresholder import Thresholder
from Rationale_Analysis.models.thresholders.global_objective import max_limited_min
from allennlp.models.model import Model

@Model.register("global_top_k")
class GlobalTopKThresholder(Thresholder) :
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
        top_indices = max_limited_min(attentions, [len(s) for s in sentences], self._max_length_ratio, self._min_inst_ratio)
        # can make things faster here with grouping top_indices by instance id first
        for b in range(attentions.shape[0]):
            sentence = [x.text for x in sentences[b]]
            
            top_ind = sorted([i for s,i in top_indices if s==b])
            top_vals = attentions[b][top_ind]
            
            rationales.append({
                'document' : " ".join([x for i, x in enumerate(sentence) if i in top_ind]),
                'spans' : [{'span' : (i, i+1), 'value' : float(v)} for i, v in zip(top_ind, top_vals)],
                'metadata' : None
            })

        return rationales