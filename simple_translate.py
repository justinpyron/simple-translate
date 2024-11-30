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
        print(f"K.shape = {K.shape}")  # TODO: delete after debugging
        print(f"Q.shape = {Q.shape}")  # TODO: delete after debugging
        scores = Q @ K.transpose(-2, -1) / self.dim_head**0.5
        print(f"scores.shape = {scores.shape}")  # TODO: delete after debugging
        if attention_mask is None:
            attention_mask = torch.ones_like(scores)
        elif attention_mask == "autoregressive":
            attention_mask = torch.tril(scores)
        else:
            pass
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
        attention_mask: torch.tensor = None,  # TODO: should this really be None by default?
    ) -> torch.tensor:
        x = x + self.dropout1(
            self.attention(self.layernorm_1(x), attention_mask=attention_mask)
        )
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
        attention_mask: torch.tensor,  # TODO: should this default to None?
        cross_x: torch.tensor,
    ) -> torch.tensor:
        x = x + self.dropout1(
            self.self_attention(self.layernorm_1(x), attention_mask="autoregressive")
        )
        x = x + self.dropout2(
            self.cross_attention(
                self.layernorm_2(x), attention_mask=attention_mask, cross_x=cross_x
            )
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
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.dim_embedding = dim_embedding
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dim_mlp = dim_mlp
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.token_embedder = nn.Embedding(vocab_size, dim_embedding)
        self.position_embedder = nn.Embedding(max_sequence_length, dim_embedding)
        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(dim_embedding, dim_head, num_heads, dim_mlp, dropout)
                for i in range(num_blocks)
            ]
        )
        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(dim_embedding, dim_head, num_heads, dim_mlp, dropout)
                for i in range(num_blocks)
            ]
        )
        self.layernorm = nn.LayerNorm(dim_embedding)
        self.classification_head = nn.Linear(dim_embedding, vocab_size)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, 0, 1 / torch.tensor(2 * self.dim_embedding).sqrt()
            )
            # Multiply by 2 bc token embedding and position embeddings get added

    def forward(self):
        pass


# TODO: padding mask for decoder? Don't we need to ignore padding tokens when computing loss?
# Otherwise, loss will be poluted by padding tokens.

# [Encoder] Self-attention, padding mask
# [Decoder] Self-attention, autoregressive mask
# [Decoder] Cross-attention, padding mask
