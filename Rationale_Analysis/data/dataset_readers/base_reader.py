import json
from typing import Dict, List, Tuple, Any

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.batch import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from numpy.random import RandomState
from overrides import overrides


def to_token(text):
    token = Token(text)
    token.info = {}
    return token


@DatasetReader.register("base_reader")
class BaseReader(DatasetReader):
    def __init__(
        self, token_indexers: Dict[str, TokenIndexer], human_prob: float = 1.0, lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = WhitespaceTokenizer()
        self._token_indexers = token_indexers
        self._human_prob = human_prob

    @overrides
    def _read(self, file_path):
        rs = RandomState(seed=1000)
        with open(cached_path(file_path), "r") as data_file:
            for _, line in enumerate(data_file.readlines()):
                items = json.loads(line)
                document = items["document"]
                annotation_id = items["annotation_id"]
                query = items.get("query", None)
                label = items.get("label", None)
                rationale = items.get("rationale", []) if rs.random_sample() < self._human_prob else []

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

    @overrides
    def text_to_instance(
        self,
        annotation_id: str,
        document: str,
        query: str = None,
        label: str = None,
        rationale: List[tuple] = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        document_tokens = [to_token(t.text) for t in self._tokenizer.tokenize(document)]
        human_rationale_labels = [0] * len(document_tokens)
        for s, e in rationale:
            for i in range(s, e):
                human_rationale_labels[i] = 1

        if query is not None:
            query_tokens = [to_token(t.text) for t in self._tokenizer.tokenize(query)]
        else:
            query_tokens = []

        for index_name, indexer in self._token_indexers.items():
            if hasattr(indexer, "add_token_info"):
                indexer.add_token_info(document_tokens, index_name)
                indexer.add_token_info(query_tokens, index_name)

        fields["document"] = MetadataField({"tokens": document_tokens, "reader_object": self})
        fields["query"] = MetadataField({"tokens": query_tokens})
        fields["rationale"] = ArrayField(np.array(human_rationale_labels))

        metadata = {
            "annotation_id": annotation_id,
            "human_rationale": rationale,
            "document": document,
            "label": label,
        }

        if query is not None:
            metadata["query"] = query

        fields["metadata"] = MetadataField(metadata)

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens: List[Token]):
        fields = {}
        tokens = tokens[0] + (([to_token("[DQSEP]")] + tokens[1]) if len(tokens[1]) > 0 else [])
        fields["document"] = TextField(tokens, self._token_indexers)

        return Instance(fields)

    def convert_documents_to_batch(
        self, documents: List[Tuple[List[Token], List[Token]]], vocabulary
    ) -> Dict[str, Any]:
        batch = Batch([self.convert_tokens_to_instance(tokens) for tokens in documents])
        batch.index_instances(vocabulary)
        batch = batch.as_tensor_dict()
        return batch["document"]

    def combine_document_query(self, document: List[MetadataField], query: List[MetadataField], vocabulary):
        document_tokens = [(x["tokens"], y["tokens"]) for x, y in zip(document, query)]
        return self.convert_documents_to_batch(document_tokens, vocabulary)
