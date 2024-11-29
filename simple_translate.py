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
        attention_mask: torch.tensor = None,
        cross_x: torch.tensor = None,
    ) -> torch.tensor:
        Q = self.query(x)
        K = self.key(x if cross_x is None else cross_x)
        V = self.value(x if cross_x is None else cross_x)
        scores = Q @ K.transpose(-2, -1) / self.dim_head**0.5
        attention_mask = (
            torch.ones_like(scores) if attention_mask is None else attention_mask
        )
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


class EncoderBlock:
    def __init__(self) -> None:
        pass

    def forward(
        self,
    ):
        pass


# # USAGE
# x = 1
# cross_x = 2

# # Self-attention, padding mask
# attention_mask = 1 # This will come from tokenizer
# x = forward(x, attention_mask)

# # Self-attention, autoregressive mask
# attention_mask = torch.tril(x)
# x = forward(x, attention_mask)

# # Cross-attention, autoregressive mask
# attention_mask = torch.tril(x)
# x = forward(x, attention_mask, cross_x)


# # TODO: multiheaded attention
# # TODO: MLP
# # TODO: encoder block: multiheaded self attention + MLP
# # TODO: decoder block: multiheaded self attention + multiheaded cross attention + MLP
