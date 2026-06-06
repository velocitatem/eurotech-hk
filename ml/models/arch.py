"""UrbanJEPA: city-scale latent world model (spec §13-16).

Tensor conventions throughout:
    B = batch, L = context time steps, T = target time steps,
    Lc = condition time steps, Z = zones, d = d_model
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.data.feature_spec import N_STATE, N_COND, N_STATIC, DECODER_DOMAINS


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
    """PatchTST-style: fold T into non-overlapping patches → transformer → mean pool.

    Handles variable-length sequences (different number of patches per call).
    """
    def __init__(self, d: int, patch_size: int, max_patches: int, heads: int, layers: int) -> None:
        super().__init__()
        self.P = patch_size
        self.patch_proj = nn.Linear(d * patch_size, d)
        self.pos_embed  = nn.Embedding(max_patches, d)
        self.tf         = _TransformerStack(d, heads, layers)
        self.norm       = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, d]  where N = B*Z
        N, T, d = x.shape
        P = self.P
        n = T // P
        x = x[:, :n * P].reshape(N, n, P * d)
        x = self.patch_proj(x)
        x = x + self.pos_embed(torch.arange(n, device=x.device))
        x = self.tf(x)
        return self.norm(x.mean(1))   # [N, d]


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------

class CityEncoder(nn.Module):
    """Encodes observed city-state windows.

    Input:  values  [B, L, Z, N_STATE]  (z-score normalised)
            masks   [B, L, Z, N_STATE]  bool, True = observed
            quality [B, L, Z, N_STATE]  float 0-1
    Output: z       [B, Z, d]
    """
    def __init__(self, d: int, n_zones: int, patch_size: int, max_patches: int,
                 heads: int, layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(N_STATE * 3, d)   # (value, mask, quality) per feature
        self.zone_embed = nn.Embedding(n_zones, d)
        self.temporal   = _PatchEncoder(d, patch_size, max_patches, heads, layers)

    def forward(self, values: torch.Tensor, masks: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        B, L, Z, _ = values.shape
        x = torch.cat([values, masks.float(), quality], dim=-1)   # [B, L, Z, N_STATE*3]
        x = self.input_proj(x)                                     # [B, L, Z, d]
        x = x + self.zone_embed(torch.arange(Z, device=x.device))
        x = x.permute(0, 2, 1, 3).reshape(B * Z, L, -1)          # [B*Z, L, d]
        z = self.temporal(x)                                        # [B*Z, d]
        return z.reshape(B, Z, -1)                                  # [B, Z, d]


# ---------------------------------------------------------------------------
# condition encoder
# ---------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """Encodes future conditions.

    Input:  cond [B, Lc, Z, N_COND]
    Output: z    [B, Z, d]
    """
    def __init__(self, d: int, n_zones: int, heads: int, layers: int) -> None:
        super().__init__()
        self.proj       = nn.Linear(N_COND, d)
        self.zone_embed = nn.Embedding(n_zones, d)
        self.tf         = _TransformerStack(d, heads, layers)
        self.norm       = nn.LayerNorm(d)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        B, Lc, Z, _ = cond.shape
        x = self.proj(cond)
        x = x + self.zone_embed(torch.arange(Z, device=x.device))
        x = x.permute(0, 2, 1, 3).reshape(B * Z, Lc, -1)
        x = self.tf(x).mean(1)     # [B*Z, d]
        return self.norm(x).reshape(B, Z, -1)


# ---------------------------------------------------------------------------
# predictor
# ---------------------------------------------------------------------------

class LatentPredictor(nn.Module):
    """Fuses context + condition latents → predicted future latent.

    Input:  z_ctx  [B, Z, d]
            z_cond [B, Z, d]
    Output: z_pred [B, Z, d]
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, z_ctx: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(torch.cat([z_ctx, z_cond], dim=-1)))


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------

class DecoderHeads(nn.Module):
    """Domain-specific heads: z_pred [B, Z, d] → Ŷ [B, T, Z, N_STATE].

    Only features registered in DECODER_DOMAINS are decoded; remaining
    STATE_FEATURES get zero-filled (loss masked anyway on missing targets).
    """
    def __init__(self, d: int, target_len: int) -> None:
        super().__init__()
        self.T = target_len
        self.heads = nn.ModuleDict({
            domain: nn.Linear(d, target_len * len(indices))
            for domain, indices in DECODER_DOMAINS.items()
        })

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, Z, _ = z.shape
        out = torch.zeros(B, self.T, Z, N_STATE, device=z.device)
        for domain, indices in DECODER_DOMAINS.items():
            n = len(indices)
            pred = self.heads[domain](z).reshape(B, Z, self.T, n).permute(0, 2, 1, 3)
            for j, idx in enumerate(indices):
                out[:, :, :, idx] = pred[:, :, :, j]
        return out


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------

class UrbanJEPA(nn.Module):
    """City-scale JEPA latent world model.

    Training call:
        z_pred, z_tgt_sg, y_hat = model(
            values_ctx, masks_ctx, quality_ctx,
            cond_future, static,
            values_tgt, masks_tgt, quality_tgt,
        )
        loss = latent_loss(z_pred, z_tgt_sg) + λ * masked_mae(y_hat, values_tgt, masks_tgt)
        model.ema_update(decay)   # after optimizer.step()

    Inference:
        z_pred, None, y_hat = model(values_ctx, masks_ctx, quality_ctx, cond_future, static)
    """

    def __init__(
        self,
        d_model:     int = 256,
        n_zones:     int = 8,
        patch_size:  int = 6,
        max_patches: int = 64,
        enc_heads:   int = 4,
        enc_layers:  int = 3,
        cond_heads:  int = 4,
        cond_layers: int = 2,
        target_len:  int = 24,
    ) -> None:
        super().__init__()
        enc_kw = dict(d=d_model, n_zones=n_zones, patch_size=patch_size,
                      max_patches=max_patches, heads=enc_heads, layers=enc_layers)
        self.city_encoder   = CityEncoder(**enc_kw)
        self.target_encoder = copy.deepcopy(self.city_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.cond_encoder = ConditionEncoder(d_model, n_zones, cond_heads, cond_layers)
        self.predictor    = LatentPredictor(d_model)
        self.decoder      = DecoderHeads(d_model, target_len)

    @torch.no_grad()
    def ema_update(self, decay: float = 0.996) -> None:
        for op, tp in zip(self.city_encoder.parameters(), self.target_encoder.parameters()):
            tp.data.mul_(decay).add_(op.data, alpha=1.0 - decay)

    def forward(
        self,
        values_ctx:  torch.Tensor,
        masks_ctx:   torch.Tensor,
        quality_ctx: torch.Tensor,
        cond_future: torch.Tensor,
        static:      torch.Tensor,           # [B, Z, N_STATIC] — hook for graph encoder
        values_tgt:  torch.Tensor | None = None,
        masks_tgt:   torch.Tensor | None = None,
        quality_tgt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        z_ctx  = self.city_encoder(values_ctx, masks_ctx, quality_ctx)
        z_cond = self.cond_encoder(cond_future)
        z_pred = self.predictor(z_ctx, z_cond)
        y_hat  = self.decoder(z_pred)

        z_tgt_sg: torch.Tensor | None = None
        if values_tgt is not None:
            q = quality_tgt if quality_tgt is not None else torch.ones_like(values_tgt)
            m = masks_tgt   if masks_tgt   is not None else torch.ones_like(values_tgt, dtype=torch.bool)
            with torch.no_grad():
                z_tgt_sg = self.target_encoder(values_tgt, m, q).detach()

        return z_pred, z_tgt_sg, y_hat
