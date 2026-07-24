"""
Evo2 embedding extraction (ArcInstitute/evo2).

Uses the evo2.Evo2 API (Vortex/StripedHyena inference) to compute mean-pooled
embeddings.

The correct layer for sequence-level representations in StripedHyena is the
final RMSNorm (`norm`) — the normalised residual stream before the unembedding
projection, analogous to last_hidden_state in BERT/GPT.  Earlier runs that used
blocks.N.mlp.l3 or blocks.N produced all-zero results because (a) `mlp.l3`
does not exist in the StripedHyena MLP, and (b) the MLP output projection is
zero-initialised in the base checkpoint, so those activations start near-zero.
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.embedders.embedding_cache import get_device as _get_device
from src.embedders.fm_embedder import FMEmbedder

# The StripedHyena module that holds the normalised residual stream,
# equivalent to last_hidden_state in encoder models.
_DEFAULT_LAYER = "norm"

_SUPPORTED_POOLINGS = {"mean", "max", "mean_max", "attention"}


def _patch_fp8_for_device():
    """Disable FP8 autocast when the GPU doesn't support it (cc < 8.9).

    All Evo2 configs set use_fp8_input_projections=True, but FP8 requires
    compute capability ≥ 8.9 (H100/A100-class). On cc 8.6 (e.g. RTX 3080 Ti)
    we force enabled=False so inference falls back to bfloat16.
    """
    if not torch.cuda.is_available():
        return
    cc = torch.cuda.get_device_capability()
    if cc >= (8, 9):
        return

    try:
        import transformer_engine.pytorch as te

        _orig_fp8_autocast = te.fp8_autocast

        from contextlib import contextmanager

        @contextmanager
        def _noop_fp8_autocast(*args, **kwargs):
            kwargs["enabled"] = False
            with _orig_fp8_autocast(*args, **kwargs):
                yield

        te.fp8_autocast = _noop_fp8_autocast
        # Also patch inside vortex layers if already imported
        try:
            import vortex.model.layers as _vl
            _vl.te.fp8_autocast = _noop_fp8_autocast
        except Exception:
            pass
        print(f"  [evo2] GPU cc={cc[0]}.{cc[1]} < 8.9 — FP8 disabled, using bfloat16 fallback.")
    except Exception as e:
        print(f"  [evo2] FP8 patch skipped: {e}")


def _get_pad_id(tokenizer) -> int:
    for attr in ("pad_token_id", "pad_id", "pad_idx"):
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            if val is not None:
                return int(val)
    return 0


def _build_mask(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Build a (B, L, 1) validity mask over the valid token positions."""
    if hidden.dim() != 3:
        raise RuntimeError(f"Unexpected Evo2 embedding shape: {tuple(hidden.shape)}")
    device = hidden.device
    lengths = lengths.to(device)
    max_len = hidden.shape[1]
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    return mask.unsqueeze(-1).to(hidden.dtype)


class _Evo2Base:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_length: int | None = None,
        device: str | None = None,
    ):
        _patch_fp8_for_device()
        from evo2 import Evo2

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.device = torch.device(device) if device else _get_device()

        print(f"Loading Evo2 {model_name} ...")
        self.evo2 = Evo2(model_name)

        # Convert model to bfloat16 to save memory
        self.evo2.model = self.evo2.model.to(torch.bfloat16)
        self.evo2.model.eval()
        self.tokenizer = self.evo2.tokenizer
        self.pad_id = _get_pad_id(self.tokenizer)

    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
        pooling = getattr(self, "pooling", "mean")
        return os.path.join(self.cache_dir, safe_model, f"pooling_{pooling}", safe_ds, f"{split}.npz")

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
    """Extract and cache mean-pooled embeddings from Evo2.

    Uses the final RMSNorm layer (`norm`) by default — the normalised residual
    stream before the unembedding head.  Pass `layer_name` to override.
    """

    def __init__(
        self,
        model_name: str = "evo2_7b_base",
        cache_dir: str = "cache/fm_embeddings",
        layer_name: str | None = None,
        max_length: int | None = None,
        device: str | None = None,
        pooling: str = "mean",
    ):
        super().__init__(model_name, cache_dir, max_length=max_length, device=device)
        if pooling not in _SUPPORTED_POOLINGS:
            raise ValueError(
                f"pooling must be one of {sorted(_SUPPORTED_POOLINGS)} "
                "(CLS pooling is not supported for Evo2: StripedHyena is "
                "autoregressive and has no special-token convention)."
            )
        self.pooling = pooling
        self.layer_name = layer_name or _DEFAULT_LAYER
        # Validate that the layer exists in the model
        try:
            self.evo2.model.get_submodule(self.layer_name)
        except AttributeError:
            available = [n for n, _ in self.evo2.model.named_modules() if n][:20]
            raise ValueError(
                f"Layer '{self.layer_name}' not found in Evo2 model. "
                f"Available (first 20): {available}"
            )
        print(f"Evo2 embedding layer: {self.layer_name}")

    def embed_sequences(
        self,
        sequences: np.ndarray,
        dataset_name: str,
        split: str,
        batch_size: int = 4,
    ) -> np.ndarray:
        import gc

        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []
        current_batch_size = max(1, batch_size)
        start = 0
        progress = tqdm(total=len(sequences), desc=f"Evo2 [{dataset_name}/{split}]")

        while start < len(sequences):
            end = min(start + current_batch_size, len(sequences))
            batch_seqs = sequences[start:end].tolist()

            try:
                input_ids, lengths = self._tokenize_batch(batch_seqs)
                input_ids = input_ids.to(self.device)

                with torch.inference_mode():
                    _, embeddings = self.evo2(
                        input_ids, return_embeddings=True, layer_names=[self.layer_name]
                    )
                hidden = embeddings[self.layer_name]
                mask = _build_mask(hidden, lengths)
                pooled = FMEmbedder.pool_hidden(hidden, mask, self.pooling)
                all_embs.append(pooled.cpu().to(torch.float16).numpy())
                progress.update(len(batch_seqs))
                start = end

            except (torch.OutOfMemoryError, RuntimeError) as exc:
                message = str(exc).lower()
                is_oom = "out of memory" in message or "cuda out of memory" in message
                if not is_oom or self.device.type != "cuda" or current_batch_size == 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                next_batch_size = max(1, current_batch_size // 2)
                print(
                    f"  [warn] CUDA OOM at batch_size={current_batch_size}; "
                    f"retrying with batch_size={next_batch_size}"
                )
                current_batch_size = next_batch_size

        progress.close()
        embeddings_arr = np.concatenate(all_embs, axis=0).astype(np.float16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings_arr)
        return embeddings_arr
