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
        # TODO: add these arguments? token_id_bos, token_id_eos, token_id_pad?

        self.token_embedder = nn.Embedding(vocab_size, dim_embedding)
        self.position_embedder = nn.Embedding(max_sequence_length, dim_embedding)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(dim_embedding, dim_head, num_heads, dim_mlp, dropout)
                for i in range(num_blocks)
            ]
        )
        self.layernorm1 = nn.LayerNorm(dim_embedding)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(dim_embedding, dim_head, num_heads, dim_mlp, dropout)
                for i in range(num_blocks)
            ]
        )
        self.layernorm2 = nn.LayerNorm(dim_embedding)
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

    def forward(
        self,
        tokens_source: torch.tensor,
        tokens_destination: torch.tensor,
        attention_mask_source: torch.tensor,
        attention_mask_destination: torch.tensor = None,
        target: torch.tensor = None,
    ):  # TODO: add typehint for output (it'll be a Union)

        # TODO: a better design would probably be this: input args = tokens_source, tokens_destination. Compute attention masks by comparing against pad_token + compute target from tokens_destination
        attention_mask_encoder = attention_mask_source.unsqueeze(dim=1).expand(
            -1, tokens_source.shape[-1], -1
        )
        attention_mask_decoder = attention_mask_source.unsqueeze(dim=1).expand(
            -1, tokens_destination.shape[-1], -1
        )
        position_idx_source = torch.arange(tokens_source.shape[1])
        position_idx_destination = torch.arange(tokens_destination.shape[1])

        # Encoder
        x_encoder = self.token_embedder(tokens_source) + self.position_embedder(
            position_idx_source
        )
        for encoder_block in self.encoder_blocks:
            x_encoder = encoder_block(x_encoder, attention_mask_encoder)
        x_encoder = self.layernorm1(x_encoder)

        # Decoder
        x_decoder = self.token_embedder(tokens_destination) + self.position_embedder(
            position_idx_destination
        )
        for decoder_block in self.decoder_blocks:
            x_decoder = decoder_block(x_decoder, attention_mask_decoder, x_encoder)
        x_decoder = self.layernorm2(x_decoder)

        # Classification head
        logits = self.classification_head(x_decoder)
        if target is None:
            return logits
        else:
            B, L, C = logits.shape
            loss_unreduced = F.cross_entropy(
                logits.view(B * L, C), target.flatten(), reduction="none"
            )
            loss = loss_unreduced.dot(attention_mask_destination.flatten().float())
            return loss.item()
