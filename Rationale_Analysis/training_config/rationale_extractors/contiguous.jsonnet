{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'max_length',
        max_length_ratio: std.extVar('MAX_LENGTH_RATIO')
    },
}