"""
Embedding layer extraction: mean-pooled token embeddings (pre-transformer).

Extracts vectors from the FM's embedding look-up table BEFORE the
transformer/Hyena layers, capturing purely static token representations
without contextual refinement.

Output: (N, embed_dim) float16, mean-pooled over non-padding tokens.
Results are cached to disk as .npz.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from src.embedders.embedding_cache import get_device as _get_device

_AUTOMODEL_PREFIXES = ("LongSafari/hyenadna",)
_CHAR_LEVEL_PREFIXES = ("LongSafari/hyenadna",)


def _is_automodel(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _AUTOMODEL_PREFIXES)


def _is_char_level(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _CHAR_LEVEL_PREFIXES)


class TokenizerEmbedder:
    """Extract mean-pooled embeddings from the embedding layer only (no transformer)."""

    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_650M_pre",
        cache_dir: str = "cache/tok_embeddings",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = _get_device()
        self._char_level = _is_char_level(model_name)

        print(f"Loading tokenizer + embedding layer for {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Load the full model, then extract only the embedding layer
        if _is_automodel(model_name):
            full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            full_model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            )

        # Locate the embedding layer
        self.embed_layer = self._find_embedding_layer(full_model)
        self.embed_layer.eval()
        self.embed_layer.to(self.device)

        # Release the rest of the model to free memory
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.embed_dim = self.embed_layer.embedding_dim
        print(f"Embedding layer ready: vocab={self.embed_layer.num_embeddings}, "
              f"dim={self.embed_dim}")

    def _find_embedding_layer(self, model) -> torch.nn.Embedding:
        """Trova l'embedding layer nel modello."""
        # NTv3 (EsmForMaskedLM): model.esm.embeddings.word_embeddings
        # HyenaDNA: model.backbone.embeddings.word_embeddings
        candidates = [
            # NTv3 / ESM
            "esm.embeddings.word_embeddings",
            # HyenaDNA
            "backbone.embeddings.word_embeddings",
            # Generic transformers
            "embeddings.word_embeddings",
            "transformer.wte",
            "model.embed_tokens",
        ]
        for attr_path in candidates:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, torch.nn.Embedding):
                    print(f"  Found embedding layer at: {attr_path}")
                    return obj
            except AttributeError:
                continue

        # Fallback: search recursively
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                print(f"  Found embedding layer at: {name}")
                return module

        raise RuntimeError(f"Cannot find embedding layer in {type(model)}")

    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
        return os.path.join(self.cache_dir, safe_model, safe_ds, f"{split}.npz")

    def _tokenize(self, batch_seqs: list[str]) -> torch.Tensor:
        if self._char_level:
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        else:
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=False,
                padding=True,
                pad_to_multiple_of=128,
                return_tensors="pt",
            )
        return tokens["input_ids"]

    def embed_sequences(
        self,
        sequences: np.ndarray,
        dataset_name: str,
        split: str,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Return (N, embed_dim) float16 array of mean-pooled embedding-layer outputs.

        Molto più veloce dell'embedding FM completo: solo un lookup + mean pool.
        """
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []
        pad_id = self.tokenizer.pad_token_id

        for start in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"TokEmb [{dataset_name}/{split}]",
        ):
            batch_seqs = sequences[start : start + batch_size].tolist()
            input_ids = self._tokenize(batch_seqs).to(self.device)

            with torch.no_grad():
                token_embs = self.embed_layer(input_ids)  # (B, L, D)

            # Mean-pool over non-padding tokens
            mask = (input_ids != pad_id).unsqueeze(-1).float()  # (B, L, 1)
            pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embs.append(pooled.cpu().to(torch.float16).numpy())

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings
