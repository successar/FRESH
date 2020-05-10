from Rationale_Analysis.models.thresholders.base_thresholder import Thresholder
from allennlp.models.model import Model
import math
import numpy as np

@Model.register("top_k")
class TopKThresholder(Thresholder) :
    def __init__(self, max_length_ratio: float) :
        self._max_length_ratio = max_length_ratio
        super().__init__()

    def forward(self, attentions, document, metadata=None) :
        rationales = self.extract_rationale(attentions=attentions, document=document)
        output_dict = {'metadata' : metadata, 'rationale' : rationales}
        return output_dict
 
    def extract_rationale(self, attentions, document, as_one_hot=False):
        attentions = attentions.cpu().data.numpy()
        document_tokens = [x["tokens"] for x in document]

        assert len(attentions) == len(document)
        assert attentions.shape[1] == max([len(d['tokens']) for d in document])

        rationales = []
        for b in range(attentions.shape[0]):
            sentence = [x.text for x in document_tokens[b]]
            attn = attentions[b][:len(sentence)]
            max_length = math.ceil(len(sentence) * self._max_length_ratio)
            
            top_ind, top_vals = np.argsort(attn)[-max_length:], np.sort(attn)[-max_length:]
            if as_one_hot :
                rationales.append([1 if i in top_ind else 0 for i in range(attentions.shape[1])])
                continue
            
            rationales.append({
                'document' : " ".join([x for i, x in enumerate(sentence) if i in top_ind]),
                'spans' : [{'span' : (i, i+1), 'value' : float(v)} for i, v in zip(top_ind, top_vals)],
            })

        return rationales