"""
Foundation model embedding extraction via HuggingFace Transformers.

Supported public models:
  - NTv3      (`InstaDeepAI/NTv3_650M_pre`)
  - HyenaDNA  (`LongSafari/hyenadna-medium-160k-seqlen-hf`)
  - DNABERT-2 (`zhihan1996/DNABERT-2-117M`)
  - Caduceus-Ph (`kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`)

Produces pooled sequence-level embeddings with disk caching in `.npz`.
"""

import gc
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from src.embedders.embedding_cache import get_device as _get_device

SUPPORTED_FM_MODELS = (
    "InstaDeepAI/NTv3_650M_pre",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "zhihan1996/DNABERT-2-117M",
    "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    "google/enformer",
)

# Models that use AutoModel (encoder backbone only, no LM head)
_AUTOMODEL_PREFIXES = (
    "LongSafari/hyenadna",
    "zhihan1996/DNABERT-2",
    "kuleshov-group/caduceus",
)

# Models that need char-level tokenization (no padding to multiple-of-128)
_CHAR_LEVEL_PREFIXES = (
    "LongSafari/hyenadna",
    "zhihan1996/DNABERT-2",
    "kuleshov-group/caduceus",
)


def _is_automodel(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _AUTOMODEL_PREFIXES)


def _is_char_level(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _CHAR_LEVEL_PREFIXES)


def validate_supported_model(model_name: str) -> None:
    if model_name in SUPPORTED_FM_MODELS:
        return
    # Also accept any model whose prefix is in the AutoModel list (e.g. new Caduceus variants)
    if _is_automodel(model_name):
        return
    supported = ", ".join(SUPPORTED_FM_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")


class FMEmbedder:
    """Extract and cache pooled embeddings from a pre-trained FM."""

    # DNABERT-2 ALiBi matrix grows as O(heads * seqlen^2); cap at 4096 tokens
    _MAX_LENGTH: dict[str, int] = {
        "zhihan1996/DNABERT-2-117M": 4096,
    }
    _MAX_LENGTH_FALLBACK = 32768

    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_650M_pre",
        cache_dir: str = "cache/fm_embeddings",
        pooling: str = "mean",
    ):
        validate_supported_model(model_name)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.pooling = pooling
        self.device = _get_device()
        self._char_level = _is_char_level(model_name)
        self.max_length = self._MAX_LENGTH.get(model_name, self._MAX_LENGTH_FALLBACK)

        if self.pooling not in {"mean", "max", "cls"}:
            raise ValueError("pooling must be one of: mean, max, cls")
        if self.pooling == "cls" and self._char_level:
            raise ValueError(
                "CLS pooling is not supported for char-level models (HyenaDNA/DNABERT-2 char mode)."
            )

        print(f"Loading model {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # bfloat16 on CUDA (NTv3 internally only autocasts to bfloat16); fp32 on CPU/MPS.
        # DNABERT-2 custom attention layers mix dtypes under bfloat16 → force fp32.
        _force_fp32 = _is_automodel(model_name) and "DNABERT" in model_name
        load_dtype = torch.float32 if (_force_fp32 or self.device.type != "cuda") else torch.bfloat16

        if _is_automodel(model_name):
            # DNABERT-2: BertConfig in older checkpoints lacks pad_token_id;
            # inject it from the tokenizer before model init to avoid AttributeError.
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
                cfg.pad_token_id = self.tokenizer.pad_token_id or 0
            # DNABERT-2's custom ALiBi init creates tensors on the default device.
            # transformers 5.x unconditionally wraps model __init__ in
            # torch.device("meta") context (see get_init_context), which crashes
            # DNABert-2's ALiBi build.  Monkey-patch to replace "meta" with "cpu".
            # DNABERT-2's bundled flash_attn_triton.py uses tl.dot(trans_b=True)
            # removed in newer Triton versions.  Force the PyTorch fallback by
            # setting attention_probs_dropout_prob > 0 in config (bert_layers.py
            # line 161: `if self.p_dropout or flash_attn_qkvpacked_func is None`).
            if not getattr(cfg, "attention_probs_dropout_prob", 0):
                cfg.attention_probs_dropout_prob = 1e-8  # negligible but non-zero

            from transformers import PreTrainedModel
            # get_init_context exists only in transformers >= 5.x (wraps __init__ in
            # meta-device context). Monkey-patch only when present; older transformers
            # don't do the meta-device wrapping so the fix isn't needed.
            _has_init_ctx = hasattr(PreTrainedModel, "get_init_context")
            if _has_init_ctx:
                _orig_get_init_context = PreTrainedModel.get_init_context

                @classmethod
                def _cpu_init_context(cls_, *args, **kwargs):
                    ctxs = _orig_get_init_context.__func__(cls_, *args, **kwargs)
                    return [torch.device("cpu") if isinstance(c, torch.device) and c.type == "meta" else c for c in ctxs]

                PreTrainedModel.get_init_context = _cpu_init_context
            # Caduceus was written against transformers 4.x; transformers 5.x added
            # two incompatible calls in _finalize_model_loading:
            #   1. model.all_tied_weights_keys  (Caduceus only has _tied_weights_keys)
            #   2. model.tie_weights(missing_keys=...) (Caduceus.tie_weights takes no kwargs)
            # Patch _finalize_model_loading to tolerate these for Caduceus models.
            _is_caduceus = model_name.startswith("kuleshov-group/caduceus")
            _orig_finalize = None
            if _is_caduceus:
                import transformers.modeling_utils as _mu
                _orig_finalize = _mu.PreTrainedModel._finalize_model_loading

                def _caduceus_finalize(model_, load_config_, loading_info_):
                    # Add all_tied_weights_keys shim if missing (transformers 5.x compat)
                    if not hasattr(model_, "all_tied_weights_keys"):
                        model_.all_tied_weights_keys = getattr(model_, "_tied_weights_keys", {}) or {}
                    # Wrap tie_weights on the instance to swallow unexpected kwargs
                    _real_tw = model_.tie_weights
                    def _safe_tw(**kw):
                        kw.pop("missing_keys", None)
                        kw.pop("recompute_mapping", None)
                        return _real_tw(**kw)
                    model_.tie_weights = _safe_tw
                    try:
                        return _orig_finalize(model_, load_config_, loading_info_)
                    finally:
                        model_.tie_weights = _real_tw

                _mu.PreTrainedModel._finalize_model_loading = staticmethod(_caduceus_finalize)
            try:
                self.model = AutoModel.from_pretrained(
                    model_name, config=cfg, trust_remote_code=True,
                    use_safetensors=True, torch_dtype=load_dtype,
                )
            finally:
                if _has_init_ctx:
                    PreTrainedModel.get_init_context = _orig_get_init_context
                if _is_caduceus and _orig_finalize is not None:
                    import transformers.modeling_utils as _mu
                    _mu.PreTrainedModel._finalize_model_loading = _orig_finalize
        else:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if self.device.type == "cuda":
                cfg.embedding_compute_dtype = "bfloat16"
                cfg.stem_compute_dtype = "bfloat16"
                cfg.down_convolution_compute_dtype = "bfloat16"
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, config=cfg, trust_remote_code=True, torch_dtype=load_dtype,
            )
        self.model.eval()
        self.model.to(self.device)
        if self.device.type == "cuda" and not _force_fp32:
            self.model.bfloat16()
        print("Model ready.")

    # ------------------------------------------------------------------
    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_model = self.model_name.replace("/", "__")
        safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
        return os.path.join(
            self.cache_dir,
            safe_model,
            f"pooling_{self.pooling}",
            safe_ds,
            f"{split}.npz",
        )

    # ------------------------------------------------------------------
    def _tokenize(self, batch_seqs: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of sequences and return tokenizer tensors on CPU."""
        add_special_tokens = self.pooling == "cls"
        if self._char_level:
            # HyenaDNA: char-level, no padding to multiple-of-128
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=add_special_tokens,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # NTv3: 6-mer tokenization, pad to multiple of 128
            tokens = self.tokenizer(
                batch_seqs,
                add_special_tokens=add_special_tokens,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                pad_to_multiple_of=128,
                return_tensors="pt",
            )
        return tokens

    # ------------------------------------------------------------------
    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass, returns last hidden state (B, L, D)."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            if _is_automodel(self.model_name):
                kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
                out = self.model(input_ids, **kwargs)
                # DNABERT-2 returns a tuple; HyenaDNA/Caduceus return an object with last_hidden_state
                if isinstance(out, (tuple, list)):
                    hidden = out[0]
                else:
                    hidden = out.last_hidden_state
            else:
                # NTv3: hidden_states[-1] is the last transformer layer output
                kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
                out = self.model(input_ids, output_hidden_states=True, **kwargs)
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
        Return (N, embed_dim) float32 array of pooled embeddings.

        Results are cached to disk as compressed .npz (float32).
        Cache path includes model name to avoid collisions between FMs.
        """
        path = self._cache_path(dataset_name, split)
        if os.path.exists(path):
            try:
                return np.load(path)["embeddings"]
            except Exception:
                import warnings
                warnings.warn(f"Corrupted cache file {path}, deleting and re-embedding.")
                os.remove(path)

        all_embs: list[np.ndarray] = []
        pad_id = self.tokenizer.pad_token_id

        start = 0
        current_batch_size = max(1, batch_size)
        progress = tqdm(total=len(sequences), desc=f"FM [{dataset_name}/{split}]")

        while start < len(sequences):
            end = min(start + current_batch_size, len(sequences))
            batch_seqs = sequences[start:end].tolist()

            try:
                tokens = self._tokenize(batch_seqs)
                input_ids = tokens["input_ids"]
                attn_mask = tokens.get("attention_mask")

                hidden = self._forward(input_ids, attention_mask=attn_mask)  # (B, L, D)

                input_ids_dev = input_ids.to(self.device)
                if attn_mask is not None:
                    mask = attn_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
                elif pad_id is not None:
                    mask = (input_ids_dev != pad_id).unsqueeze(-1).to(hidden.dtype)
                else:
                    mask = torch.ones_like(hidden[:, :, :1], dtype=hidden.dtype)

                if self.pooling == "mean":
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                elif self.pooling == "max":
                    neg_inf = torch.finfo(hidden.dtype).min
                    masked_hidden = hidden.masked_fill(mask == 0, neg_inf)
                    pooled = masked_hidden.max(dim=1).values
                    empty_rows = (mask.sum(dim=1).squeeze(-1) == 0)
                    if empty_rows.any():
                        pooled[empty_rows] = 0
                else:  # cls
                    pooled = hidden[:, 0, :]

                all_embs.append(pooled.cpu().float().numpy())
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
                if next_batch_size == current_batch_size:
                    raise

                print(
                    f"  [warn] CUDA OOM at batch_size={current_batch_size}; "
                    f"retrying with batch_size={next_batch_size}"
                )
                current_batch_size = next_batch_size

        progress.close()

        embeddings = np.concatenate(all_embs, axis=0).astype(np.float16)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, embeddings=embeddings)
        return embeddings
