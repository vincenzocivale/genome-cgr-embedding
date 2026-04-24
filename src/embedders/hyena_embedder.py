"""
HyenaDNA embedding extraction via HuggingFace Transformers.

HyenaDNA is a long-range genomics model (up to 160k bp) based on structured
convolutions (Hyena operator). Produces mean-pooled sequence-level embeddings
with disk caching in compressed .npz format.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.embedders.embedding_cache import get_device as _get_device


class HyenaEmbedder:
    """Extract and cache mean-pooled embeddings from HyenaDNA."""

    def __init__(
        self,
        model_name: str = "LongSafari/hyenadna-medium-160k-seqlen-hf",
        cache_dir: str = "cache/hyena_embeddings",
        pooling: str = "mean",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.pooling = pooling
        self.device = _get_device()

        if self.pooling not in {"mean", "max"}:
            raise ValueError("HyenaEmbedder supports only pooling in {'mean', 'max'}")

        print(f"Loading HyenaDNA {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.model.eval()
        self.model.to(self.device)
        print("Model ready.")

    # ------------------------------------------------------------------
    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe = dataset_name.replace("/", "__").replace("\\", "__")
        return os.path.join(self.cache_dir, f"pooling_{self.pooling}", safe, f"{split}.npz")

    # ------------------------------------------------------------------
    def embed_sequences(
        self,
        sequences: np.ndarray,
        dataset_name: str,
        split: str,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Return (N, embed_dim) float32 array of pooled embeddings.

        HyenaDNA usa single-char tokenization (A,C,G,T).
        Output: hidden state medio su tutti i token non-padding.

        Results are cached to disk as compressed .npz (float32).
        """
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            return np.load(path)["embeddings"]

        all_embs: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"HyenaDNA [{dataset_name}/{split}]",
        ):
            batch_seqs = sequences[start : start + batch_size].tolist()

            # HyenaDNA uses single-char tokenization
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                out = self.model(**tokens, output_hidden_states=True)

            # HyenaDNA returns hidden_states as a tuple
            # Take the last hidden state
            if hasattr(out, "hidden_states") and out.hidden_states:
                hidden = out.hidden_states[-1]  # (B, L, D)
            else:
                # Fallback: use last_hidden_state
                hidden = out.last_hidden_state  # (B, L, D)

            # Mean-pool over non-padding tokens
            # HyenaDNA tokenizer does not return attention_mask; build it from pad_token_id
            pad_id = self.tokenizer.pad_token_id
            mask = (tokens["input_ids"] != pad_id).unsqueeze(-1).float()  # (B, L, 1)
            if self.pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
            else:
                neg_inf = torch.finfo(hidden.dtype).min
                masked_hidden = hidden.masked_fill(mask == 0, neg_inf)
                pooled = masked_hidden.max(dim=1).values
                empty_rows = (mask.sum(dim=1).squeeze(-1) == 0)
                if empty_rows.any():
                    pooled[empty_rows] = 0
            all_embs.append(pooled.cpu().float().numpy())

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings
