from __future__ import annotations

import torch

from ml.data.feature_spec import STATE_IDX
from ml.models.arch import UrbanJEPA
from ml.models.train import masked_mae


def test_text_conditioning_learns_directional_reaction() -> None:
    torch.manual_seed(7)

    batch_size = 2
    context_len = 8
    target_len = 4
    n_zones = 2
    text_dim = 16

    model = UrbanJEPA(
        d_model=16,
        n_zones=n_zones,
        patch_size=2,
        max_patches=8,
        enc_heads=4,
        enc_layers=1,
        text_dim=text_dim,
        target_len=target_len,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    values_ctx = torch.zeros(batch_size, context_len, n_zones, 17)
    masks_ctx = torch.ones_like(values_ctx, dtype=torch.bool)
    quality_ctx = torch.ones_like(values_ctx)
    static = torch.zeros(batch_size, n_zones, 5)

    text_emb = torch.zeros(batch_size, text_dim)
    text_emb[1, :4] = torch.tensor([1.0, -0.5, 0.75, 0.25])

    values_tgt = torch.zeros(batch_size, target_len, n_zones, 17)
    masks_tgt = torch.zeros_like(values_tgt, dtype=torch.bool)

    increased_features = [
        "traffic_intensity",
        "traffic_occupancy",
        "parking_occupancy",
        "air_quality_index",
        "energy_demand",
    ]
    decreased_features = ["traffic_speed"]

    for name in increased_features:
        idx = STATE_IDX[name]
        values_tgt[1, :, :, idx] = 1.5
        masks_tgt[:, :, :, idx] = True

    for name in decreased_features:
        idx = STATE_IDX[name]
        values_tgt[1, :, :, idx] = -1.0
        masks_tgt[:, :, :, idx] = True

    model.train()
    for _ in range(250):
        _, _, y_hat = model(values_ctx, masks_ctx, quality_ctx, text_emb, static)
        loss = masked_mae(y_hat, values_tgt, masks_tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        _, _, y_base = model(
            values_ctx[:1],
            masks_ctx[:1],
            quality_ctx[:1],
            text_emb[:1],
            static[:1],
        )
        _, _, y_event = model(
            values_ctx[:1],
            masks_ctx[:1],
            quality_ctx[:1],
            text_emb[1:2],
            static[:1],
        )

    delta = (y_event - y_base)[0]

    for name in increased_features:
        idx = STATE_IDX[name]
        assert float(delta[:, :, idx].mean()) > 0.5, name

    for name in decreased_features:
        idx = STATE_IDX[name]
        assert float(delta[:, :, idx].mean()) < -0.3, name
