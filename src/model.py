import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, num_items: int, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1, max_seq_len: int = 50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name:
                nn.init.normal_(p, mean=0, std=0.01)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, item_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = item_seq.shape
        positions = torch.arange(seq_len, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1
        ).bool()
        pad_mask = (item_seq == 0)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.layer_norm(x)

        last_idx = (lengths - 1).clamp(min=0)
        return x[torch.arange(batch_size), last_idx]

    def predict_scores(self, user_repr: torch.Tensor) -> torch.Tensor:
        return user_repr @ self.item_emb.weight.T
