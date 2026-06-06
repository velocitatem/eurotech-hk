"""UrbanJEPA: city-scale latent world model.

JEPA dynamics:
    S_future = f(S_past, A_transition)

where A_transition is a free-form text description of events, policies,
shocks, or planned activity during the transition window.

    S_context ──▶ CityEncoder      ──▶ z_ctx
    text/news  ──▶ TextCondEncoder  ──▶ a_text
    z_ctx + a_text ──▶ Predictor   ──▶ z_pred
    z_pred ──▶ Decoder             ──▶ ŷ_state

Target encoder is an EMA copy of CityEncoder (no-grad, updated after each step).
Text embeddings are pre-computed offline via Qwen3-Embedding-0.6B (text_dim=1024).
During initial training with no labelled text, pass zeros — the model learns to be
informative when text is present and degrades gracefully when absent.

Tensor conventions:
    B = batch, L = context steps, T = target steps,
    Z = zones, d = d_model, E = text_dim
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.data.feature_spec import N_STATE, N_STATIC, DECODER_DOMAINS


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------

class _TransformerStack(nn.Module):
    def __init__(self, d: int, heads: int, layers: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d, heads, d * ff_mult, dropout=0.1, batch_first=True, norm_first=True
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _PatchEncoder(nn.Module):
    """PatchTST-style: fold T into non-overlapping patches → transformer → mean pool."""
    def __init__(self, d: int, patch_size: int, max_patches: int, heads: int, layers: int) -> None:
        super().__init__()
        self.P          = patch_size
        self.patch_proj = nn.Linear(d * patch_size, d)
        self.pos_embed  = nn.Embedding(max_patches, d)
        self.tf         = _TransformerStack(d, heads, layers)
        self.norm       = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, d = x.shape
        P = self.P
        n = T // P
        x = x[:, :n * P].reshape(N, n, P * d)
        x = self.patch_proj(x)
        x = x + self.pos_embed(torch.arange(n, device=x.device))
        return self.norm(self.tf(x).mean(1))   # [N, d]


# ---------------------------------------------------------------------------
# city encoder (shared between online + EMA target branch)
# ---------------------------------------------------------------------------

class CityEncoder(nn.Module):
    """Encodes observed city-state windows → latent [B, Z, d].

    Input:  values  [B, L, Z, N_STATE]
            masks   [B, L, Z, N_STATE]  bool — True = observed
            quality [B, L, Z, N_STATE]  float 0-1
    """
    def __init__(self, d: int, n_zones: int, patch_size: int, max_patches: int,
                 heads: int, layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(N_STATE * 3, d)
        self.zone_embed = nn.Embedding(n_zones, d)
        self.temporal   = _PatchEncoder(d, patch_size, max_patches, heads, layers)

    def forward(self, values: torch.Tensor, masks: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        B, L, Z, _ = values.shape
        x = torch.cat([values, masks.float(), quality], dim=-1)   # [B, L, Z, N_STATE*3]
        x = self.input_proj(x)
        x = x + self.zone_embed(torch.arange(Z, device=x.device))
        x = x.permute(0, 2, 1, 3).reshape(B * Z, L, -1)
        z = self.temporal(x)                                        # [B*Z, d]
        return z.reshape(B, Z, -1)                                  # [B, Z, d]


# ---------------------------------------------------------------------------
# text condition encoder
# ---------------------------------------------------------------------------

class TextConditionEncoder(nn.Module):
    """Projects pre-computed sentence embeddings → latent action token.

    Input:  text_emb [B, text_dim]   — sentence embedding (zeros = no-text)
    Output: a_text   [B, d_model]    — single global action vector

    The action token is global (one per batch item, not per zone). The predictor
    broadcasts it across zones before fusion so each zone sees the same event
    context but combines it with its own latent trajectory.
    """
    def __init__(self, text_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(text_emb))   # [B, d]


# ---------------------------------------------------------------------------
# predictor
# ---------------------------------------------------------------------------

class LatentPredictor(nn.Module):
    """Fuses per-zone context latent with global text action → predicted latent.

    z_ctx:  [B, Z, d]  — online encoder output
    a_text: [B, d]     — text condition encoder output (global)
    →       [B, Z, d]  — predicted target latent
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, z_ctx: torch.Tensor, a_text: torch.Tensor) -> torch.Tensor:
        a = a_text.unsqueeze(1).expand_as(z_ctx)   # [B, d] → broadcast [B, Z, d]
        return self.norm(self.mlp(torch.cat([z_ctx, a], dim=-1)))


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------

class DecoderHeads(nn.Module):
    """Domain heads: z_pred [B, Z, d] → Ŷ [B, T, Z, N_STATE]."""
    def __init__(self, d: int, target_len: int) -> None:
        super().__init__()
        self.T     = target_len
        self.heads = nn.ModuleDict({
            domain: nn.Linear(d, target_len * len(indices))
            for domain, indices in DECODER_DOMAINS.items()
        })

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, Z, _ = z.shape
        out = torch.zeros(B, self.T, Z, N_STATE, device=z.device)
        for domain, indices in DECODER_DOMAINS.items():
            n    = len(indices)
            pred = self.heads[domain](z).reshape(B, Z, self.T, n).permute(0, 2, 1, 3)
            for j, idx in enumerate(indices):
                out[:, :, :, idx] = pred[:, :, :, j]
        return out


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------

class UrbanJEPA(nn.Module):
    """City-scale JEPA world model conditioned on free-form text transitions.

    Training:
        z_pred, z_tgt_sg, y_hat = model(
            values_ctx, masks_ctx, quality_ctx, text_emb, static,
            values_tgt, masks_tgt,
        )
        loss = latent_loss(z_pred, z_tgt_sg) + λ * masked_mae(y_hat, values_tgt, masks_tgt)
        model.ema_update(decay)

    Inference (no target):
        z_pred, None, y_hat = model(values_ctx, masks_ctx, quality_ctx, text_emb, static)
    """

    def __init__(
        self,
        d_model:     int = 256,
        n_zones:     int = 8,
        patch_size:  int = 6,
        max_patches: int = 64,
        enc_heads:   int = 4,
        enc_layers:  int = 3,
        text_dim:    int = 1024,
        target_len:  int = 24,
    ) -> None:
        super().__init__()
        enc_kw = dict(d=d_model, n_zones=n_zones, patch_size=patch_size,
                      max_patches=max_patches, heads=enc_heads, layers=enc_layers)
        self.city_encoder   = CityEncoder(**enc_kw)
        self.target_encoder = copy.deepcopy(self.city_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder = TextConditionEncoder(text_dim, d_model)
        self.predictor    = LatentPredictor(d_model)
        self.decoder      = DecoderHeads(d_model, target_len)

    @torch.no_grad()
    def ema_update(self, decay: float = 0.996) -> None:
        for op, tp in zip(self.city_encoder.parameters(), self.target_encoder.parameters()):
            tp.data.mul_(decay).add_(op.data, alpha=1.0 - decay)

    def forward(
        self,
        values_ctx:  torch.Tensor,          # [B, L, Z, N_STATE]
        masks_ctx:   torch.Tensor,          # [B, L, Z, N_STATE] bool
        quality_ctx: torch.Tensor,          # [B, L, Z, N_STATE]
        text_emb:    torch.Tensor,          # [B, text_dim]  — zeros if no text
        static:      torch.Tensor,          # [B, Z, N_STATIC]
        values_tgt:  torch.Tensor | None = None,
        masks_tgt:   torch.Tensor | None = None,
        quality_tgt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        z_ctx  = self.city_encoder(values_ctx, masks_ctx, quality_ctx)
        a_text = self.text_encoder(text_emb)
        z_pred = self.predictor(z_ctx, a_text)
        y_hat  = self.decoder(z_pred)

        z_tgt_sg: torch.Tensor | None = None
        if values_tgt is not None:
            q = quality_tgt if quality_tgt is not None else torch.ones_like(values_tgt)
            m = masks_tgt   if masks_tgt   is not None else torch.ones_like(values_tgt, dtype=torch.bool)
            with torch.no_grad():
                z_tgt_sg = self.target_encoder(values_tgt, m, q).detach()

        return z_pred, z_tgt_sg, y_hat
