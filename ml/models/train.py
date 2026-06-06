"""JEPA training loop for UrbanJEPA (spec §25).

Loss:
    L_total = L_latent(cosine_dist) + λ_raw * L_raw(masked_MAE) + λ_reg * L2_reg
EMA target encoder is updated after every optimizer step.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dlib import get_logger
from ml.models.arch import UrbanJEPA

logger = get_logger("ml-train")


def latent_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()


def masked_mae(y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.float().sum().clamp(min=1.0)
    return ((y_hat - y_true).abs() * mask.float()).sum() / n


class UrbanWindowDataset(Dataset):
    """Wraps a split dict from dataset.pt into a torch Dataset."""

    _keys = ("values_ctx", "masks_ctx", "quality_ctx", "cond_future",
             "static", "values_tgt", "masks_tgt")

    def __init__(self, split: dict[str, torch.Tensor]) -> None:
        self.data = {k: split[k] for k in self._keys}

    def __len__(self) -> int:
        return len(self.data["values_ctx"])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        return tuple(self.data[k][i] for k in self._keys)


class JEPATrainer:
    def __init__(
        self,
        model:      UrbanJEPA,
        loader:     DataLoader,
        lr:         float,
        lambda_raw: float,
        lambda_reg: float,
        ema_decay:  float,
        log_dir:    str,
        log_every:  int,
        device:     torch.device,
    ) -> None:
        self.model      = model.to(device)
        self.loader     = loader
        self.opt        = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.lam_raw    = lambda_raw
        self.lam_reg    = lambda_reg
        self.ema_decay  = ema_decay
        self.writer     = SummaryWriter(log_dir)
        self.log_every  = log_every
        self.device     = device
        self.step       = 0

    def _to(self, *ts: torch.Tensor) -> list[torch.Tensor]:
        return [t.to(self.device) for t in ts]

    def _l2_reg(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.model.parameters() if p.requires_grad)  # type: ignore[return-value]

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in self.loader:
            v_ctx, m_ctx, q_ctx, cond, static, v_tgt, m_tgt = self._to(*batch)
            z_pred, z_tgt, y_hat = self.model(
                v_ctx, m_ctx.bool(), q_ctx, cond, static,
                v_tgt, m_tgt.bool(),
            )
            l_lat = latent_loss(z_pred, z_tgt)          # type: ignore[arg-type]
            l_raw = masked_mae(y_hat, v_tgt, m_tgt.bool())
            l_reg = self._l2_reg()
            loss  = l_lat + self.lam_raw * l_raw + self.lam_reg * l_reg

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.model.ema_update(self.ema_decay)

            total += loss.item()
            if self.step % self.log_every == 0:
                self.writer.add_scalar("train/loss_latent", l_lat.item(), self.step)
                self.writer.add_scalar("train/loss_raw",    l_raw.item(), self.step)
                self.writer.add_scalar("train/loss_total",  loss.item(),  self.step)
            self.step += 1
        return total / max(len(self.loader), 1)

    def train(self, epochs: int) -> None:
        for epoch in range(epochs):
            avg = self.train_epoch()
            self.writer.add_scalar("train/epoch_loss", avg, epoch)
            logger.info(f"epoch={epoch + 1}/{epochs} loss={avg:.5f}")
        self.writer.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="ml/configs/train/default.yaml")
    p.add_argument("--dataset", default="ml/data/processed/dataset.pt")
    p.add_argument("--weights", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["seed"]))

    blob    = torch.load(args.dataset, map_location="cpu", weights_only=False)
    ds      = UrbanWindowDataset(blob["train"])
    loader  = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                         num_workers=0, pin_memory=device.type == "cuda")

    model = UrbanJEPA(
        d_model=int(cfg["d_model"]),
        n_zones=int(cfg["n_zones"]),
        patch_size=int(cfg["patch_size"]),
        max_patches=int(cfg["max_patches"]),
        enc_heads=int(cfg["enc_heads"]),
        enc_layers=int(cfg["enc_layers"]),
        cond_heads=int(cfg["cond_heads"]),
        cond_layers=int(cfg["cond_layers"]),
        target_len=int(cfg["target_len"]),
    )
    logger.info(f"params={sum(p.numel() for p in model.parameters()):,}")

    trainer = JEPATrainer(
        model=model,
        loader=loader,
        lr=float(cfg["learning_rate"]),
        lambda_raw=float(cfg["lambda_raw"]),
        lambda_reg=float(cfg.get("lambda_reg", 1e-5)),
        ema_decay=float(cfg["ema_decay"]),
        log_dir=str(cfg["tensorboard_dir"]),
        log_every=int(cfg["log_every_n_steps"]),
        device=device,
    )
    trainer.train(int(cfg["epochs"]))

    out = Path(args.weights or cfg.get("weights_output", "ml/models/weights/model.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    logger.info(f"saved_weights={out}")


if __name__ == "__main__":
    main()
