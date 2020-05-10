local bert_type = std.extVar('BERT_TYPE');

local base_model = {
  bert_model: bert_type,
  requires_grad: '10,11,pooler',
  dropout : 0.2,
};

local base_model_movies = {
  text_field_embedder: {
    token_embedders: {
      bert: {
        type: "pretrained_transformer_mismatched",
        model_name: bert_type,
        max_length: 512
      },
    },
  },
  seq2seq_encoder: {
    type: "lstm",
    input_size: 768,
    hidden_size: 128,
    bidirectional: true
  },
  dropout: 0.2,
  requires_grad: '10,11,pooler',
  feedforward_encoder:{
    input_dim: 256,
    num_layers: 1,
    hidden_dims: [128],
    activations: ['relu'],
    dropout: 0.2
  },
};

local base_indexer = {
  model_name: bert_type
};

local indexer = "pretrained-simple";
local indexer_movies = "pretrained-simple-movies";

local bert_model = base_model + {
  type: "bert_classifier",
};

local bert_model_movies = base_model_movies + {
  type: "bert_lstm_classifier"
};

local bert_gen_model = base_model + {
  type: "bernoulli_bert_generator",
};

local bert_gen_model_movies = base_model_movies + {
  type: "bernoulli_bert_lstm_generator"
};

local bert_extractor_model = base_model + {
  type: "supervised_bert_extractor",
};

local bert_extractor_model_movies = base_model_movies + {
  type: "supervised_bert_lstm_extractor"
};

local is_movies = if std.findSubstr('movies', std.extVar('TRAIN_DATA_PATH')) == [] then false else true;


{
    classifier: if is_movies then bert_model_movies else bert_model,
    generator: if is_movies then bert_gen_model_movies else bert_gen_model,
    indexer: base_indexer + {
      type : if is_movies then indexer_movies else indexer
    } + (if is_movies then {max_length: 512} else {}),
    extractor: if is_movies then bert_extractor_model_movies else bert_extractor_model
}