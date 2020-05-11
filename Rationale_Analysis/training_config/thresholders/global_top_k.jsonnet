{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'global_top_k',
        max_length_ratio: std.parseJson(std.extVar('MAX_LENGTH_RATIO')),
        min_inst_ratio: std.parseJson(std.extVar('MIN_INST_RATIO')),
    },
}