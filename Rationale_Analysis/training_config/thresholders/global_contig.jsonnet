{
    dataset_reader : {
        type : "saliency_reader",
    },
    validation_dataset_reader : {
        type : "saliency_reader",
    },
    model : {
        type : 'global_contig',
        max_length_ratio: std.parseInt(std.extVar('MAX_LENGTH_PERCENT')) / 100,
        min_inst_ratio: std.parseInt(std.extVar('MIN_INST_PERCENT')) / 100,
    },
}