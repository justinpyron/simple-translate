from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim_head = dim_head
        self.query = nn.Linear(dim_embedding, dim_head, bias=False)
        self.key = nn.Linear(dim_embedding, dim_head, bias=False)
        self.value = nn.Linear(dim_embedding, dim_head, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
        attention_mask: Union[str, torch.tensor] = None,
        cross_x: torch.tensor = None,
    ) -> torch.tensor:
        Q = self.query(x)
        K = self.key(x if cross_x is None else cross_x)
        V = self.value(x if cross_x is None else cross_x)
        scores = Q @ K.transpose(-2, -1) / self.dim_head**0.5
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = attention_weights @ V
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_embedding, dim_head, dropout) for i in range(num_heads)]
        )
        self.linear = nn.Linear(dim_head * num_heads, dim_embedding)

    def forward(
        self,
        x: torch.tensor,
        attention_mask: torch.tensor = None,
        cross_x: torch.tensor = None,
    ) -> torch.tensor:
        x = torch.cat([head(x, attention_mask, cross_x) for head in self.heads], dim=-1)
        x = self.linear(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
        dim_mlp: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Self-attention (with padding mask) layer
        self.layernorm_1 = nn.LayerNorm(dim_embedding)
        self.attention = MultiHeadedAttention(
            dim_embedding, dim_head, num_heads, dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        # Feed-forward layer
        self.layernorm_2 = nn.LayerNorm(dim_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_embedding),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
        attention_mask: torch.tensor,
    ) -> torch.tensor:
        x = x + self.dropout1(self.attention(self.layernorm_1(x), attention_mask))
        x = x + self.dropout2(self.mlp(self.layernorm_2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
        dim_mlp: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Self-attention (with autoregressive mask) layer
        self.layernorm_1 = nn.LayerNorm(dim_embedding)
        self.self_attention = MultiHeadedAttention(
            dim_embedding, dim_head, num_heads, dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        # Cross-attention (with padding mask) layer
        self.layernorm_2 = nn.LayerNorm(dim_embedding)
        self.cross_attention = MultiHeadedAttention(
            dim_embedding, dim_head, num_heads, dropout
        )
        self.dropout2 = nn.Dropout(dropout)
        # Feed-forward layer
        self.layernorm_3 = nn.LayerNorm(dim_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_embedding),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
        attention_mask: torch.tensor,
        cross_x: torch.tensor,
    ) -> torch.tensor:
        n_examples, n_tokens, _ = x.shape
        autoregressive_mask = torch.tril(torch.ones(n_examples, n_tokens, n_tokens))
        x = x + self.dropout1(
            self.self_attention(self.layernorm_1(x), autoregressive_mask)
        )
        x = x + self.dropout2(
            self.cross_attention(self.layernorm_2(x), attention_mask, cross_x)
        )
        x = x + self.dropout3(self.mlp(self.layernorm_3(x)))
        return x


class SimpleTranslate(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        dim_embedding: int,
        dim_head: int,
        num_heads: int,
        dim_mlp: int,
        dropout: float,
        num_blocks: int,
        token_id_bos: int,
        token_id_eos: int,  # TODO: review if this is necessary
        token_id_pad: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.dim_embedding = dim_embedding
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dim_mlp = dim_mlp
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.token_id_bos = token_id_bos
        self.token_id_eos = token_id_eos
        self.token_id_pad = token_id_pad
        self.token_embedder = nn.Embedding(vocab_size, dim_embedding)
        self.position_embedder = nn.Embedding(max_sequence_length, dim_embedding)
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(dim_embedding, dim_head, num_heads, dim_mlp, dropout)
                for i in range(num_blocks)
            ]
        )
        self.layernorm_encoder = nn.LayerNorm(dim_embedding)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(dim_embedding, dim_head, num_heads, dim_mlp, dropout)
                for i in range(num_blocks)
            ]
        )
        self.layernorm_decoder = nn.LayerNorm(dim_embedding)
        self.classification_head = nn.Linear(dim_embedding, vocab_size)
        self.apply(self.init_weights)

    def init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0,
                std=1 / torch.tensor(2 * self.dim_embedding).sqrt(),
                # Multiply by 2 bc token and position embeddings get added
            )

    def forward_encoder(
        self,
        tokens_source: torch.tensor,
        attention_mask_encoder: torch.tensor,
    ) -> torch.tensor:
        position_idx_source = torch.arange(tokens_source.shape[1])
        x_encoder = self.token_embedder(tokens_source) + self.position_embedder(
            position_idx_source
        )
        for block in self.encoder:
            x_encoder = block(x_encoder, attention_mask_encoder)
        x_encoder = self.layernorm_encoder(x_encoder)
        return x_encoder

    def forward_decoder(
        self,
        tokens_destination: torch.tensor,
        attention_mask_decoder: torch.tensor,
        x_encoder: torch.tensor,
    ) -> torch.tensor:
        position_idx_destination = torch.arange(tokens_destination.shape[1])
        x_decoder = self.token_embedder(tokens_destination) + self.position_embedder(
            position_idx_destination
        )
        for block in self.decoder:
            x_decoder = block(x_decoder, attention_mask_decoder, x_encoder)
        x_decoder = self.layernorm_decoder(x_decoder)
        return x_decoder

    def forward(
        self,
        tokens_source: torch.tensor,
        tokens_destination: torch.tensor = None,
    ) -> Union[torch.tensor, float]:
        # Training vs inference processing
        if tokens_destination is None:
            targets = None
            tokens_destination = torch.full(
                (tokens_source.shape[0], 1), self.token_id_bos
            )
        else:
            targets = tokens_destination[:, 1:]
            tokens_destination = tokens_destination[:, :-1]
        # Attention masks
        pad_mask_source = (tokens_source != self.token_id_pad).int()
        pad_mask_destination = (tokens_destination != self.token_id_pad).float()
        attention_mask_encoder = pad_mask_source.unsqueeze(dim=1).expand(
            -1, tokens_source.shape[-1], -1
        )
        attention_mask_decoder = pad_mask_source.unsqueeze(dim=1).expand(
            -1, tokens_destination.shape[-1], -1
        )
        # Forward pass
        x_encoder = self.forward_encoder(tokens_source, attention_mask_encoder)
        x_decoder = self.forward_decoder(
            tokens_destination, attention_mask_decoder, x_encoder
        )
        logits = self.classification_head(x_decoder)
        # Output
        if targets is None:
            return logits
        else:
            batch_size, n_tokens, n_classes = logits.shape
            logits_flat = logits.view(batch_size * n_tokens, n_classes)
            targets_flat = targets.flatten()
            loss_unreduced = F.cross_entropy(
                logits_flat, targets_flat, reduction="none"
            )
            loss = (
                loss_unreduced * pad_mask_destination.flatten()
            ).mean()  # Ignore padding tokens inside loss function
            return loss.item()
