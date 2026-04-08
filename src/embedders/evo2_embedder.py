"""
Evo2 embedding extraction (ArcInstitute/evo2).

Uses the evo2.Evo2 API (Vortex inference) to compute mean-pooled embeddings.
Supports layer-hook extraction for FM embeddings and direct embedding-layer
lookups for tokenizer-only embeddings.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.embedders.embedding_cache import get_device as _get_device


def _get_pad_id(tokenizer) -> int:
    for attr in ("pad_token_id", "pad_id", "pad_idx"):
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            if val is not None:
                return int(val)
    return 0


def _choose_default_layer(model: torch.nn.Module) -> str:
    """Pick a reasonable default layer name for Evo2 embeddings."""
    names = [name for name, _ in model.named_modules()]

    # Prefer the last MLP output within the transformer blocks if present.
    candidates: list[tuple[int, str]] = []
    for name in names:
        m = re.match(r"blocks\.(\d+)\.mlp\.l3$", name)
        if m:
            candidates.append((int(m.group(1)), name))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]

    # Fallback: last block output.
    candidates = []
    for name in names:
        m = re.match(r"blocks\.(\d+)$", name)
        if m:
            candidates.append((int(m.group(1)), name))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]

    # Last-resort: any module containing "block"
    for name in reversed(names):
        if "block" in name.lower():
            return name

    raise RuntimeError(
        "Cannot auto-select an Evo2 layer. "
        "Please pass --evo2-layer with a valid module name."
    )


def _mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool over sequence length using token lengths."""
    if hidden.dim() == 2:
        return hidden
    if hidden.dim() != 3:
        raise RuntimeError(f"Unexpected Evo2 embedding shape: {tuple(hidden.shape)}")

    device = hidden.device
    lengths = lengths.to(device)
    max_len = hidden.shape[1]
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


class _Evo2Base:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_length: int | None = None,
        device: str | None = None,
    ):
        # Import locally to avoid hard dependency when Evo2 is not used.
        from evo2 import Evo2

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.device = torch.device(device) if device else _get_device()

        print(f"Loading Evo2 {model_name} ...")
        self.evo2 = Evo2(model_name)
        self.evo2.model.eval()
        self.tokenizer = self.evo2.tokenizer
        self.pad_id = _get_pad_id(self.tokenizer)

    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
        return os.path.join(self.cache_dir, safe_model, safe_ds, f"{split}.npz")

    def _tokenize_batch(self, sequences: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        token_lists: list[list[int]] = []
        lengths: list[int] = []
        for seq in sequences:
            ids = list(self.tokenizer.tokenize(str(seq)))
            if self.max_length is not None and len(ids) > self.max_length:
                ids = ids[: self.max_length]
            token_lists.append(ids)
            lengths.append(len(ids))

        max_len = max(lengths) if lengths else 0
        input_ids = torch.full(
            (len(token_lists), max_len),
            fill_value=self.pad_id,
            dtype=torch.long,
        )
        for i, ids in enumerate(token_lists):
            if ids:
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        return input_ids, torch.tensor(lengths, dtype=torch.long)


class Evo2Embedder(_Evo2Base):
    """Extract and cache mean-pooled embeddings from Evo2."""

    def __init__(
        self,
        model_name: str = "evo2_7b",
        cache_dir: str = "cache/fm_embeddings",
        layer_name: str | None = None,
        max_length: int | None = None,
        device: str | None = None,
    ):
        super().__init__(model_name, cache_dir, max_length=max_length, device=device)
        self.layer_name = layer_name or _choose_default_layer(self.evo2.model)
        print(f"Evo2 embedding layer: {self.layer_name}")

    def embed_sequences(
        self,
        sequences: np.ndarray,
        dataset_name: str,
        split: str,
        batch_size: int = 4,
    ) -> np.ndarray:
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Evo2 [{dataset_name}/{split}]",
        ):
            batch_seqs = sequences[start : start + batch_size].tolist()
            input_ids, lengths = self._tokenize_batch(batch_seqs)
            input_ids = input_ids.to(self.device)

            with torch.no_grad():
                _, embeddings = self.evo2(
                    input_ids, return_embeddings=True, layer_names=[self.layer_name]
                )
            hidden = embeddings[self.layer_name]
            pooled = _mean_pool(hidden, lengths)
            all_embs.append(pooled.detach().cpu().to(torch.float16).numpy())

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings


class Evo2TokenizerEmbedder(_Evo2Base):
    """Extract mean-pooled embeddings from the token embedding layer only."""

    def __init__(
        self,
        model_name: str = "evo2_7b",
        cache_dir: str = "cache/tok_embeddings",
        max_length: int | None = None,
        device: str | None = None,
    ):
        super().__init__(model_name, cache_dir, max_length=max_length, device=device)
        self.embed_layer = self._find_embedding_layer(self.evo2.model)
        self.embed_layer.eval()
        self.embed_layer.to(self.device)
        print(
            f"Evo2 token embedding layer: "
            f"vocab={self.embed_layer.num_embeddings}, "
            f"dim={self.embed_layer.embedding_dim}"
        )

    @staticmethod
    def _find_embedding_layer(model: torch.nn.Module) -> torch.nn.Embedding:
        candidates: list[tuple[str, torch.nn.Embedding]] = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                candidates.append((name, module))

        if not candidates:
            raise RuntimeError("No torch.nn.Embedding layers found in Evo2 model.")

        # Prefer names that look like token embeddings.
        for key in ("token", "word", "embed"):
            for name, module in candidates:
                if key in name.lower():
                    return module

        # Fallback to the first embedding found.
        return candidates[0][1]

    def embed_sequences(
        self,
        sequences: np.ndarray,
        dataset_name: str,
        split: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Evo2Tok [{dataset_name}/{split}]",
        ):
            batch_seqs = sequences[start : start + batch_size].tolist()
            input_ids, lengths = self._tokenize_batch(batch_seqs)
            input_ids = input_ids.to(self.device)

            with torch.no_grad():
                token_embs = self.embed_layer(input_ids)

            pooled = _mean_pool(token_embs, lengths)
            all_embs.append(pooled.detach().cpu().to(torch.float16).numpy())

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings
