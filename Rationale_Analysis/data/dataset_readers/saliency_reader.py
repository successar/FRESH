import json
from typing import Dict, Any
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance

from allennlp.data.tokenizers import Token


@DatasetReader.register("saliency_reader")
class SaliencyReader(DatasetReader):
    def __init__(self, lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                items = json.loads(line)
                saliency = np.array(items["saliency"])
                document = items["document"]
                metadata = {k: v for k, v in items.items() if k != "saliency"}
                instance = self.text_to_instance(saliency=saliency, document=document, metadata=metadata)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, saliency, document, metadata: Dict[str, Any]) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}
        fields["attentions"] = ArrayField(saliency, padding_value=0.0)
        fields["document"] = MetadataField({"tokens": [Token(t) for t in document.split()]})
        fields["metadata"] = MetadataField(metadata)

        assert len(saliency) == len(fields["document"].metadata["tokens"])

        return Instance(fields)
