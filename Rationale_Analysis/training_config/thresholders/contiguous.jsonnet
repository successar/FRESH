{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'contiguous',
        max_length_ratio: std.parseJson(std.extVar('MAX_LENGTH_RATIO'))
    },
}