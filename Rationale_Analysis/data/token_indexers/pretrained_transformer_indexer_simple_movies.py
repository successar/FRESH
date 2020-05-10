import logging
from typing import Dict, List, Tuple

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


@TokenIndexer.register("pretrained-simple-movies")
class PretrainedTransformerIndexerSimpleMovies(PretrainedTransformerIndexer):
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:

        tokens = get_tokens(tokens)

        token_wordpiece_ids = [
            [token.info[self._index_name]["wordpiece-ids"] for token in token_list]
            for token_list in tokens
        ]

        assert len(tokens) == 1

        wordpiece_ids, type_ids, offsets_doc = self.intra_word_tokenize_sentence(token_wordpiece_ids[0])

        token_mask = [True]*len(tokens[0])

        output_dict = {
            "token_ids" : wordpiece_ids,
            "wordpiece_mask": [True] * len(wordpiece_ids),
            "mask" : token_mask,
            "type_ids": type_ids,
            "offsets": offsets_doc
        }

        return self._postprocess_output(output_dict)

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

        tokens = tokens.copy()
        padding_lengths = padding_lengths.copy()

        offsets_tokens = tokens.pop("offsets")
        offsets_padding_lengths = padding_lengths.pop("offsets")

        tensor_dict = {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val,
                    padding_lengths[key],
                    default_value=lambda: 0
                    if "mask" in key or "type-ids" in key
                    else self._tokenizer.pad_token_id,
                )
            )
            for key, val in tokens.items()
        }

        tensor_dict["offsets"] = torch.LongTensor(
            pad_sequence_to_length(
                offsets_tokens, offsets_padding_lengths, default_value=lambda: (0, 0)
            )
        )

        return tensor_dict

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

    def intra_word_tokenize_sentence(
        self, tokens_a: List[List[int]]
    ) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:

        wordpieces_a, offsets_a, cumulative = self.intra_word_tokenize_in_id(
            tokens_a, self._allennlp_tokenizer.num_added_start_tokens
        )

        text_ids = self._tokenizer.build_inputs_with_special_tokens(wordpieces_a)
        type_ids = self._tokenizer.create_token_type_ids_from_sequences(wordpieces_a)

        assert cumulative + self._allennlp_tokenizer.num_added_end_tokens == len(text_ids)
        return text_ids, type_ids, offsets_a
