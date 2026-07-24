"""
Shared utilities for device detection and embedding cache I/O.

All embedder classes and ridge scripts use the same logic for detecting the
compute device and loading pre-computed embeddings from disk.
"""

import os

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    env_dev = os.environ.get("CGR_DEVICE") or os.environ.get("GENOME_CGR_DEVICE")
    if env_dev:
        return torch.device(env_dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cached_embeddings(
    model_name: str,
    dataset_name: str,
    split: str,
    pooling: str = "mean",
) -> np.ndarray | None:
    """Load embeddings from cache (.npz) if available, otherwise return None.

    Handles two cache structures produced by the embedder classes:
      - FMEmbedder / TokenizerEmbedder:
            cache/fm_embeddings/{model_tag}/{dataset}/{split}.npz
      - HyenaEmbedder:
            cache/hyena_embeddings/{dataset}/{split}.npz  (no model sub-dir)

    Returns float32 array of shape (N, embed_dim), or None if not cached.
    """
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")

    candidates = [
        os.path.join(
            "cache/fm_embeddings",
            safe_model,
            f"pooling_{pooling}",
            safe_ds,
            f"{split}.npz",
        ),
    ]
    if pooling == "mean":
        # Pre-rebuttal caches predate the pooling-aware subdirectory and
        # contain mean-pooled embeddings only; never use them as a stand-in
        # for a non-mean pooling that hasn't been computed yet.
        candidates.append(os.path.join("cache/fm_embeddings", safe_model, safe_ds, f"{split}.npz"))
        candidates.append(os.path.join("cache/fm_embeddings", safe_ds, f"{split}.npz"))
    if "hyenadna" in model_name.lower():
        candidates.append(
            os.path.join("cache/hyena_embeddings", f"pooling_{pooling}", safe_ds, f"{split}.npz")
        )
        if pooling == "mean":
            candidates.append(
                os.path.join("cache/hyena_embeddings", safe_ds, f"{split}.npz")
            )
    for path in candidates:
        if os.path.exists(path):
            return np.load(path)["embeddings"].astype(np.float32)
    return None
