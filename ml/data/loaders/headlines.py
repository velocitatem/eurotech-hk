"""HKFP headlines from R2 → per-day mean sentence embeddings for UrbanJEPA text conditioning.

Uses Qwen3-Embedding-0.6B (1024-dim) as the text encoder. Embeddings are computed once
and cached to disk. `build_text_panel` returns [T, TEXT_DIM] aligned to an hourly
DatetimeIndex — each hour gets the mean embedding of all headlines for that calendar day
(zeros when absent). Falls back to zeros if the model is unavailable.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from dlib import get_logger
from .r2_data import _r2_csv

logger   = get_logger("loader-headlines")
TEXT_DIM = 1024
_MODEL   = "Qwen/Qwen3-Embedding-0.6B"
_CACHE   = Path("ml/data/processed/.r2_cache/hk_hkfp_headlines_embeddings_qwen3.npz")


def _embed(titles: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_MODEL, trust_remote_code=True)
        return np.asarray(
            model.encode(titles, batch_size=32, show_progress_bar=True, normalize_embeddings=True),
            dtype=np.float32,
        )
    except ImportError:
        logger.warning("sentence_transformers not installed — returning zero embeddings")
        return np.zeros((len(titles), TEXT_DIM), dtype=np.float32)


def _load_or_compute(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings [N, TEXT_DIM], dates [N] str array)."""
    if _CACHE.exists():
        d = np.load(_CACHE, allow_pickle=True)
        logger.info(f"headlines: loaded {len(d['embeddings'])} cached embeddings")
        return d["embeddings"].astype(np.float32), d["dates"]

    titles = df["title"].fillna("").tolist()
    embs   = _embed(titles)
    dates  = df["date"].values
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE, embeddings=embs, dates=dates)
    logger.info(f"headlines: computed & cached {len(embs)} embeddings → {_CACHE}")
    return embs, dates


def build_text_panel(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """[T, TEXT_DIM] float32 — hourly panel aligned to `timestamps`, zeros when no headlines."""
    df = _r2_csv("hk_hkfp_headlines.csv")
    url_dates = df["url"].str.extract(r"/(\d{4})/(\d{2})/(\d{2})/").apply(
        lambda c: c.str.zfill(2)
    ).agg("-".join, axis=1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna(url_dates)
    embs, dates  = _load_or_compute(df)

    idx_df       = pd.DataFrame({"date": dates, "i": np.arange(len(dates))})
    date_to_emb  = {
        d: embs[grp["i"].values].mean(axis=0)
        for d, grp in idx_df.groupby("date")
    }

    panel = np.zeros((len(timestamps), TEXT_DIM), dtype=np.float32)
    for h, ts in enumerate(timestamps):
        key = ts.strftime("%Y-%m-%d")
        if key in date_to_emb:
            panel[h] = date_to_emb[key]
    logger.info(f"headlines: built text panel shape={panel.shape}, covered days={len(date_to_emb)}")
    return panel
