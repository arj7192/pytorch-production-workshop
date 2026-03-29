"""
Small GPT-style transformer for language modeling.

Architecture: decoder-only transformer with causal masking.
~8M parameters at default config -  small enough to train on CPU
during a workshop, large enough to demonstrate real patterns.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    """
    Decoder-only transformer language model.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Embedding and hidden dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward inner dimension.
        n_layers: Number of transformer decoder layers.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.output_proj.weight = self.token_emb.weight

        self._init_weights()
        self._causal_mask_cache = {}

    def _init_weights(self):
        """Xavier uniform initialization for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token indices.
            targets: (batch, seq_len) target token indices for loss computation.

        Returns:
            dict with 'logits' and optionally 'loss'.
        """
        seq_len = input_ids.size(1)
        causal_mask = self._get_causal_mask(seq_len, input_ids.device)

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        # Decoder-only: memory is a dummy zero tensor (self-attention only)
        memory = torch.zeros(
            input_ids.size(0), 1, self.d_model, device=input_ids.device
        )
        x = self.transformer(x, memory, tgt_mask=causal_mask)
        logits = self.output_proj(x)

        result = {"logits": logits}

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressive text generation with top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.max_seq_len :]
            output = self(idx_cond)
            logits = output["logits"][:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> TransformerLM:
    """Factory function: build a TransformerLM from a config dict."""
    return TransformerLM(
        vocab_size=config["vocab_size"],
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 4),
        max_seq_len=config.get("max_seq_len", 128),
        dropout=config.get("dropout", 0.1),
    )
