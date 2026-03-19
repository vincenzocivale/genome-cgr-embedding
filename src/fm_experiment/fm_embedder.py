"""
Foundation model embedding extraction via HuggingFace Transformers.

Supports:
  - NTv3   (InstaDeepAI/NTv3_*)          AutoModelForMaskedLM  6-mer tokenization
  - HyenaDNA (LongSafari/hyenadna-*-hf)  AutoModel             char-level tokenization

Produces mean-pooled sequence-level embeddings with disk caching in .npz (float16).
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


# Models that use AutoModel (encoder backbone only, no LM head)
_AUTOMODEL_PREFIXES = ("LongSafari/hyenadna",)

# Models that need char-level tokenization (no padding to multiple-of-128)
_CHAR_LEVEL_PREFIXES = ("LongSafari/hyenadna",)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_automodel(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _AUTOMODEL_PREFIXES)


def _is_char_level(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _CHAR_LEVEL_PREFIXES)


class FMEmbedder:
    """Extract and cache mean-pooled embeddings from a pre-trained FM."""

    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_650M_pre",
        cache_dir: str = "cache/fm_embeddings",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = _get_device()
        self._char_level = _is_char_level(model_name)

        print(f"Loading model {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if _is_automodel(model_name):
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            )
        self.model.eval()
        self.model.to(self.device)
        print("Model ready.")

    # ------------------------------------------------------------------
    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
        return os.path.join(self.cache_dir, safe_model, safe_ds, f"{split}.npz")

    # ------------------------------------------------------------------
    def _tokenize(self, batch_seqs: list[str]) -> torch.Tensor:
        """Tokenize a batch of sequences and return input_ids on CPU."""
        if self._char_level:
            # HyenaDNA: char-level, no padding to multiple-of-128
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        else:
            # NTv3: 6-mer tokenization, pad to multiple of 128
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=False,
                padding=True,
                pad_to_multiple_of=128,
                return_tensors="pt",
            )
        return tokens["input_ids"]

    # ------------------------------------------------------------------
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns last hidden state (B, L, D)."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            if _is_automodel(self.model_name):
                # HyenaDNA returns (B, L, D) directly as last_hidden_state
                out = self.model(input_ids)
                hidden = out.last_hidden_state
            else:
                # NTv3: hidden_states[-1] is the last transformer layer output
                out = self.model(input_ids, output_hidden_states=True)
                hidden = out.hidden_states[-1]
        return hidden

    # ------------------------------------------------------------------
    def embed_sequences(
        self,
        sequences: np.ndarray,
        dataset_name: str,
        split: str,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Return (N, embed_dim) float16 array of mean-pooled embeddings.

        Results are cached to disk as compressed .npz (float16).
        Cache path includes model name to avoid collisions between FMs.
        """
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []
        pad_id = self.tokenizer.pad_token_id

        for start in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"FM [{dataset_name}/{split}]",
        ):
            batch_seqs = sequences[start : start + batch_size].tolist()
            input_ids = self._tokenize(batch_seqs)

            hidden = self._forward(input_ids)  # (B, L, D)

            # Mean-pool over non-padding tokens
            input_ids_dev = input_ids.to(self.device)
            mask = (input_ids_dev != pad_id).unsqueeze(-1).float()  # (B, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
            all_embs.append(pooled.cpu().to(torch.float16).numpy())

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings
