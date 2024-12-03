from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_1000/")
model_configs = {
    "vocab_size": tokenizer.vocab_size,
    "max_sequence_length": 256,
    "dim_embedding": 128,
    "dim_head": 64,
    "num_heads": 8,
    "dim_mlp": 256,
    "dropout": 0.1,
    "num_blocks": 4,
    "token_id_bos": tokenizer.bos_token_id,
    "token_id_eos": tokenizer.eos_token_id,
    "token_id_pad": tokenizer.pad_token_id,
}

tokenizer_mini = PreTrainedTokenizerFast.from_pretrained("tokenizer_0500/")
model_configs_mini = {
    "vocab_size": tokenizer_mini.vocab_size,
    "max_sequence_length": 256,
    "dim_embedding": 64,
    "dim_head": 32,
    "num_heads": 4,
    "dim_mlp": 128,
    "dropout": 0.1,
    "num_blocks": 2,
    "token_id_bos": tokenizer_mini.bos_token_id,
    "token_id_eos": tokenizer_mini.eos_token_id,
    "token_id_pad": tokenizer_mini.pad_token_id,
}
