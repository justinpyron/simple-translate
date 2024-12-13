from collections import namedtuple
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
            if attention_mask == "autoregressive":
                attention_mask = torch.tril(scores)
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
        # LAYER 1: Self-attention (with padding mask)
        self.layernorm_1 = nn.LayerNorm(dim_embedding)
        self.attention = MultiHeadedAttention(
            dim_embedding, dim_head, num_heads, dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        # LAYER 2: Feed-forward
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
        # LAYER 1: Self-attention (with autoregressive mask)
        self.layernorm_1 = nn.LayerNorm(dim_embedding)
        self.self_attention = MultiHeadedAttention(
            dim_embedding, dim_head, num_heads, dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        # LAYER 2: Cross-attention (with padding mask)
        self.layernorm_2 = nn.LayerNorm(dim_embedding)
        self.cross_attention = MultiHeadedAttention(
            dim_embedding, dim_head, num_heads, dropout
        )
        self.dropout2 = nn.Dropout(dropout)
        # LAYER 3: Feed-forward layer
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
        x = x + self.dropout1(
            self.self_attention(self.layernorm_1(x), "autoregressive")
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
        self.register_buffer(
            "position_idx", torch.arange(max_sequence_length), persistent=False
        )
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
        n_tokens = tokens_source.shape[1]
        token_embedding = self.token_embedder(tokens_source)
        position_embedding = self.position_embedder(self.position_idx[:n_tokens])
        x_encoder = token_embedding + position_embedding
        for encoder_block in self.encoder:
            x_encoder = encoder_block(x_encoder, attention_mask_encoder)
        x_encoder = self.layernorm_encoder(x_encoder)
        return x_encoder

    def forward_decoder(
        self,
        tokens_destination: torch.tensor,
        attention_mask_decoder: torch.tensor,
        x_encoder: torch.tensor,
    ) -> torch.tensor:
        n_tokens = tokens_destination.shape[1]
        token_embedding = self.token_embedder(tokens_destination)
        position_embedding = self.position_embedder(self.position_idx[:n_tokens])
        x_decoder = token_embedding + position_embedding
        for decoder_block in self.decoder:
            x_decoder = decoder_block(x_decoder, attention_mask_decoder, x_encoder)
        x_decoder = self.layernorm_decoder(x_decoder)
        return x_decoder

    def forward(
        self,
        tokens_source: torch.tensor,
        tokens_destination: torch.tensor,
        targets: torch.tensor = None,
    ) -> Union[torch.tensor, float]:
        # STEP 1: Attention masks
        pad_mask_source = (tokens_source != self.token_id_pad).int()
        pad_mask_destination = (tokens_destination != self.token_id_pad).float()
        attention_mask_encoder = pad_mask_source.unsqueeze(dim=1).expand(
            -1, tokens_source.shape[-1], -1
        )
        attention_mask_decoder = pad_mask_source.unsqueeze(dim=1).expand(
            -1, tokens_destination.shape[-1], -1
        )
        # STEP 2: Forward pass
        x_encoder = self.forward_encoder(tokens_source, attention_mask_encoder)
        x_decoder = self.forward_decoder(
            tokens_destination, attention_mask_decoder, x_encoder
        )
        logits = self.classification_head(x_decoder)
        # STEP 3: Output
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
            return loss

    def generate(
        self,
        tokens_source: torch.tensor,
        tokens_destination: torch.tensor = None,
        temperature: float = 1e-3,
    ) -> torch.tensor:
        """
        Generate translation for a single input example.

        Input tokens are expected to be in a batch of size 1.

        Temperature contols the randomness of generated output.
        The lower the temperature, the lower the randomness.
        As temperature approaches 0, the probability of sampling
        the most likely next token approaches 1.
        """
        self.eval()
        if tokens_destination is None:
            tokens_destination = torch.tensor([[self.token_id_bos]])
        with torch.no_grad():
            for i in range(self.max_sequence_length - 1):
                # TODO: Update to keep generating until EOS. Or max_iters, which doesn't have to be max_sequence_length
                # TODO: when computing forward pass, take the most recent max_sequence_length tokens.
                logits = self.forward(tokens_source, tokens_destination)
                logits_final_token = logits[:, -1, :]
                probability = F.softmax(logits_final_token / temperature, dim=-1)
                next_token = torch.multinomial(probability, num_samples=1)
                tokens_destination = torch.cat((tokens_destination, next_token), dim=-1)
                if next_token[0, 0] == self.token_id_eos:
                    break
        return tokens_destination

    def generate_with_beams(
        self,
        tokens_source: torch.tensor,
        tokens_destination: torch.tensor = None,
        temperature: float = 1e-3,
        beam_width: int = 10,
        max_new_tokens: int = 200,
    ) -> torch.tensor:
        """
        Generate translation for a single input example using beam search.

        Input tokens are expected to be in a batch of size 1.

        Temperature contols the randomness of generated output.
        The lower the temperature, the lower the randomness.
        As temperature approaches 0, the probability of sampling
        the most likely next token approaches 1.
        """
        self.eval()
        if tokens_destination is None:
            tokens_destination = torch.tensor([[self.token_id_bos]])
        Beam = namedtuple("Beam", ["tokens", "cumulative_probability"])
        with torch.no_grad():
            beams = [Beam(tokens_destination, 1)]
            for i in range(max_new_tokens):
                new_beams = list()
                for beam in beams:
                    logits = self.forward(tokens_source, beam.tokens)[0, -1, :]
                    probability = F.softmax(logits / temperature, dim=-1)
                    candidate_tokens = probability.argsort(descending=True)[:beam_width]
                    candidate_probabilities = probability[candidate_tokens]
                    new_beams += [
                        Beam(
                            torch.cat((beam.tokens, torch.tensor([[token]])), dim=-1),
                            beam.cumulative_probability * prob,
                        )
                        for token, prob in zip(
                            candidate_tokens, candidate_probabilities
                        )
                    ]
                new_beams.sort(key=lambda x: x.cumulative_probability, reverse=True)
                beams = new_beams[:beam_width]
                # Stopping condition
                tokens = torch.vstack([beam.tokens for beam in beams])
                if (tokens == self.token_id_eos).any(dim=-1).all():
                    break
        return beams[0].tokens
