"""
Enformer embedding extraction via HuggingFace Transformers.

Uses AutoTokenizer and AutoModel to compute mean-pooled sequence embeddings.
Falls back to FMEmbedder behavior when necessary.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from tqdm import tqdm

from src.embedders.embedding_cache import get_device as _get_device

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None


class EnformerEmbedder:
    """Extract and cache mean-pooled embeddings from google/enformer.

    Produces float16 embeddings cached as compressed `.npz`.
    """

    def __init__(self, model_name: str = "google/enformer", cache_dir: str = "cache/fm_embeddings"):
        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError("transformers not available to load Enformer model")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = _get_device()

        print(f"Loading Enformer {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.eval()
        self.model.to(self.device)
        print("Enformer model ready.")

    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
        return os.path.join(self.cache_dir, safe_model, f"pooling_mean", safe_ds, f"{split}.npz")

    def embed_sequences(self, sequences: np.ndarray, dataset_name: str, split: str, batch_size: int = 16) -> np.ndarray:
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []

        for start in tqdm(range(0, len(sequences), batch_size), desc=f"Enformer [{dataset_name}/{split}]"):
            batch_seqs = sequences[start : start + batch_size].tolist()
            tokens = self.tokenizer(batch_seqs, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                out = self.model(**tokens, output_hidden_states=True)

            # prefer hidden_states if present
            if hasattr(out, "hidden_states") and out.hidden_states:
                hidden = out.hidden_states[-1]
            else:
                hidden = out.last_hidden_state

            mask = None
            if "attention_mask" in tokens:
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            else:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is not None and "input_ids" in tokens:
                    mask = (tokens["input_ids"] != pad_id).unsqueeze(-1).to(hidden.dtype)

            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            all_embs.append(pooled.cpu().float().numpy())

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings
