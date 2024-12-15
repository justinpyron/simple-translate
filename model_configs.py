from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_1000/")
model_configs = {
    "vocab_size": tokenizer.vocab_size,
    "max_sequence_length": 256,
    "dim_embedding": 128,
    "dim_head": 16,
    "num_heads": 8,
    "dim_mlp": 256,
    "dropout": 0.1,
    "num_blocks": 4,
    "token_id_bos": tokenizer.bos_token_id,
    "token_id_eos": tokenizer.eos_token_id,
    "token_id_pad": tokenizer.pad_token_id,
}
