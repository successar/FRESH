import json

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from numpy.random import RandomState
from overrides import overrides

from Rationale_Analysis.data.dataset_readers.base_reader import BaseReader


@DatasetReader.register("extractor_reader")
class ExtractorReader(BaseReader):
    @overrides
    def _read(self, file_path):
        rs = RandomState(seed=1000)
        with open(cached_path(file_path), "r") as data_file:
            for _, line in enumerate(data_file.readlines()):
                items = json.loads(line)
                document = items["original_document"]
                annotation_id = items["annotation_id"]
                query = items.get("query", None)
                label = items.get("label", None)
                if rs.random_sample() < self._human_prob:
                    rationale = items.get("human_rationale")
                else:
                    rationale = items.get("predicted_rationale")["spans"]
                    rationale = [span["span"] for span in rationale]

                if label is not None:
                    label = str(label).replace(" ", "_")

                instance = self.text_to_instance(
                    annotation_id=annotation_id,
                    document=document,
                    query=query,
                    label=label,
                    rationale=rationale,
                )
                yield instance
