from allennlp.models.model import Model
from typing import Dict, Any
import torch


class Thresholder(Model):
    def __init__(self):
        super().__init__(vocab=None)
        self._keepsake_param = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, **kwargs):
        # pylint: disable=arguments-differ

        raise NotImplementedError

    def extract_rationale(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("No Method to Extract Rationale")

    def make_output_human_readable(self, output_dict):
        new_output_dict = {}

        new_output_dict["predicted_rationale"] = output_dict["rationale"]
        new_output_dict["document"] = [r["document"] for r in output_dict["rationale"]]

        if "query" in output_dict["metadata"][0]:
            new_output_dict["query"] = [m["query"] for m in output_dict["metadata"]]

        new_output_dict["label"] = [m["label"] for m in output_dict["metadata"]]
        new_output_dict["annotation_id"] = [m["annotation_id"] for m in output_dict["metadata"]]
        new_output_dict["human_rationale"] = [m["human_rationale"] for m in output_dict["metadata"]]
        new_output_dict["original_document"] = [m["document"] for m in output_dict["metadata"]]

        return new_output_dict
