import torch
from allennlp.nn import util


def generate_embeddings_for_pooling(sequence_tensor, span_starts, span_ends):
    #(B, L, E), #(B, L), #(B, L)
    span_starts = span_starts.unsqueeze(-1)
    span_ends = (span_ends - 1).unsqueeze(-1)
    span_widths = span_ends - span_starts
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = util.get_range_vector(max_batch_span_width, util.get_device_of(sequence_tensor)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = (max_span_range_indices <= span_widths).float()
    raw_span_indices = span_ends - max_span_range_indices
    # We also don't want to include span indices which are less than zero,
    # which happens because some spans near the beginning of the sequence
    # have an end index < max_batch_span_width, so we add this to the mask here.
    span_mask = span_mask * (raw_span_indices >= 0).float()
    span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

    # Shape: (batch_size * num_spans * max_batch_span_width)
    flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

    return span_embeddings, span_mask