{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'top_k',
        max_length_ratio: std.extVar('MAX_LENGTH_RATIO')
    },
}