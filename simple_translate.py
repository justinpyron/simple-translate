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
        x_attention_mask: torch.tensor = None,
        cross_x: torch.tensor = None,
    ) -> torch.tensor:
        Q = self.query(x)
        K = self.key(x if cross_x is None else cross_x)
        V = self.value(x if cross_x is None else cross_x)
        scores = Q @ K.transpose(-2, -1) / self.dim_head**0.5
        x_attention_mask = (
            torch.ones_like(scores) if x_attention_mask is None else x_attention_mask
        )
        scores = scores.masked_fill(x_attention_mask == 0, float("-inf"))
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
        x_attention_mask: torch.tensor = None,
        cross_x: torch.tensor = None,
    ) -> torch.tensor:
        x = torch.cat(
            [head(x, x_attention_mask, cross_x) for head in self.heads], dim=-1
        )
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
        x_attention_mask: torch.tensor = None,
    ) -> torch.tensor:
        x = x + self.dropout1(self.attention(self.layernorm_1(x), x_attention_mask))
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

        # Cross-attention (with autoregressive mask) layer
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
        x_attention_mask: torch.tensor = None,
        cross_x: torch.tensor = None,
    ) -> torch.tensor:
        x = x + self.dropout1(
            self.self_attention(self.layernorm_1(x), x_attention_mask)
        )
        x = x + self.dropout2(
            self.cross_attention(self.layernorm_2(x), x_attention_mask)
        )

        x = x + self.dropout2(self.mlp(self.layernorm_2(x)))
        return x


# USAGE
# x = 1
# cross_x = 2

# [Encoder] Self-attention, padding mask
# attention_mask = 1 # This will come from tokenizer
# x = forward(x, attention_mask)

# [Decoder] Self-attention, autoregressive mask
# attention_mask = torch.tril(x)
# x = forward(x, attention_mask)

# [Decoder] Cross-attention, autoregressive mask
# attention_mask = torch.tril(x)
# x = forward(x, attention_mask, cross_x)


# # TODO: multiheaded attention
# # TODO: MLP
# # TODO: encoder block: multiheaded self attention + MLP
# # TODO: decoder block: multiheaded self attention + multiheaded cross attention + MLP
