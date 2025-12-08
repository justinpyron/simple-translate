# Strong dependency on the tokenizer in tokenizer_1000/
model_configs = {
    "vocab_size": 1000,
    "max_sequence_length": 256,
    "dim_embedding": 128,
    "dim_head": 16,
    "num_heads": 8,
    "dim_mlp": 256,
    "dropout": 0.1,
    "num_blocks": 4,
    "token_id_bos": 0,
    "token_id_eos": 1,
    "token_id_pad": 2,
}
