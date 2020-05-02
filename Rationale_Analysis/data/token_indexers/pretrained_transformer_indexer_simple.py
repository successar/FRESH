import logging
from typing import Dict, List, Tuple

from itertools import groupby
import numpy as np

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList, TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)

def get_tokens(tokens) :
    index_of_separator = set([i for i, x in enumerate(tokens) if x.text == "[DQSEP]"])
    assert len(index_of_separator) <= 1
    if len(index_of_separator) == 0 :
        tokens = [tokens]
    else :
        index_of_separator = list(index_of_separator)[0]
        tokens = [tokens[:index_of_separator], tokens[index_of_separator + 1:]]
    return tokens


@TokenIndexer.register("pretrained-simple")
class PretrainedTransformerIndexerSimple(PretrainedTransformerIndexer):
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:

        tokens = get_tokens(tokens)

        token_wordpiece_ids = [
            [token.info[self._index_name]["wordpiece-ids"] for token in token_list]
            for token_list in tokens
        ]

        if len(tokens) == 2 :
            wordpiece_ids, type_ids, attention_mask, offsets_doc, offsets_query = self.intra_word_tokenize_sentence_pair(token_wordpiece_ids[0], token_wordpiece_ids[1])
        else :
            wordpiece_ids, type_ids, attention_mask, offsets_doc = self.intra_word_tokenize_sentence(token_wordpiece_ids[0])

        if len(offsets_doc) == 0 :
            doc_starting_offsets, doc_ending_offsets = [], []
        else :
            doc_starting_offsets, doc_ending_offsets = list(zip(*offsets_doc))

        if len(wordpiece_ids) > 512:
            postions_ids = [
                i * 512 / len(wordpiece_ids) for i in range(len(wordpiece_ids))
            ]
        else:
            postions_ids = list(range(len(wordpiece_ids)))

        token_mask = [1]*len(tokens[0])
        wordpiece_mask = [1] * len(wordpiece_ids)
        wordpiece_to_tokens = [-1] * len(wordpiece_ids)
        for i, (start, end) in enumerate(zip(doc_starting_offsets, doc_ending_offsets)) :
            for j in range(start, end) :
                wordpiece_to_tokens[j] = i

        return {
            "wordpiece-ids": wordpiece_ids,
            "document-starting-offsets": list(doc_starting_offsets),
            "document-ending-offsets": list(doc_ending_offsets),
            "type-ids": type_ids,
            "position-ids": postions_ids,
            "wordpiece-mask": wordpiece_mask,
            "mask": token_mask,
            "attention-mask": list(attention_mask),
            "wordpiece-to-token" : wordpiece_to_tokens
        }

    def add_token_info(self, tokens: List[Token], index_name: str):
        self._index_name = index_name
        for token in tokens:
            wordpieces = self._tokenizer.tokenize(token.text)
            if len(wordpieces) == 0:
                token.info[index_name] = {
                    "wordpiece-ids": [self._tokenizer.unk_token_id]
                }
                continue

            token.info[index_name] = {
                "wordpiece-ids": [
                    bpe_id
                    for bpe_id in self._tokenizer.encode(
                        wordpieces, add_special_tokens=False
                    )
                ]
            }

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        # Different transformers use different padding values for tokens, but for mask and type id, the padding
        # value is always 0.
        return {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val,
                    padding_lengths[key],
                    default_value=lambda: 0 if key == 'attention-mask' else (0
                    if "mask" in key or "type-ids" in key
                    else self._tokenizer.pad_token_id),
                )
            )
            for key, val in tokens.items()
        }

    def intra_word_tokenize_in_id(
        self, tokens: List[List[int]], starting_offset: int = 0
    ) -> Tuple[List[int], List[Tuple[int, int]], int]:

        wordpieces: List[int] = []
        offsets = []

        cumulative = starting_offset

        for token in tokens:
            subword_wordpieces = token
            wordpieces.extend(subword_wordpieces)
            start_offset = cumulative
            cumulative += len(subword_wordpieces)
            end_offset = cumulative  # exclusive end offset
            offsets.append((start_offset, end_offset))

        return wordpieces, offsets, cumulative

    def intra_word_tokenize_sentence_pair(
        self, tokens_a: List[List[int]], tokens_b: List[List[int]]
    ) -> Tuple[List[int], List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:

        wordpieces_a, offsets_a, cumulative = self.intra_word_tokenize_in_id(
            tokens_a, self._allennlp_tokenizer.num_added_start_tokens
        )

        # c1 = cumulative

        wordpieces_b, offsets_b, cumulative = self.intra_word_tokenize_in_id(
            tokens_b, cumulative + self._allennlp_tokenizer.num_added_middle_tokens
        )

        text_ids = self._tokenizer.build_inputs_with_special_tokens(wordpieces_a, wordpieces_b)
        type_ids = self._tokenizer.create_token_type_ids_from_sequences(wordpieces_a, wordpieces_b)

        attention_mask = np.array([1]*len(text_ids))
        attention_mask[:self._allennlp_tokenizer.num_added_start_tokens] = 2
        attention_mask[-130:] = 2

        assert cumulative + self._allennlp_tokenizer.num_added_end_tokens == len(text_ids)
        return text_ids, type_ids, attention_mask, offsets_a, offsets_b

    def intra_word_tokenize_sentence(
        self, tokens_a: List[List[int]]
    ) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:

        wordpieces_a, offsets_a, cumulative = self.intra_word_tokenize_in_id(
            tokens_a, self._allennlp_tokenizer.num_added_start_tokens
        )

        text_ids = self._tokenizer.build_inputs_with_special_tokens(wordpieces_a)
        type_ids = self._tokenizer.create_token_type_ids_from_sequences(wordpieces_a)
        attention_mask = np.array([1]*len(text_ids))
        attention_mask[:self._allennlp_tokenizer.num_added_start_tokens] = 2
        attention_mask[-self._allennlp_tokenizer.num_added_end_tokens:] = 2

        assert cumulative + self._allennlp_tokenizer.num_added_end_tokens == len(text_ids)
        return text_ids, type_ids, attention_mask, offsets_a
