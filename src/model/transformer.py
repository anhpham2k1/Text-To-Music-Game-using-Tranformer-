"""
Music Transformer — Conditional Music Generation Model.

Kiến trúc: Transformer Decoder with:
- Token Embedding + Rotary Position Encoding (RoPE)
- N × DecoderBlock (Self-Attn RoPE → Cross-Attn → SwiGLUFFN + RMSNorm)
- Linear output projection → vocab_size

Conditioning (deep optimization):
- Structured PromptEncoder (lightweight) — recommended for training
- NLPPromptEncoder (BERT-tiny, lazy-loaded) — for free-text inference
- Supports pre-encoded cond for fast generation
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .embedding import TokenEmbedding
from .layers import DecoderBlock, RMSNorm
from .prompt_encoder import PromptEncoder, NLPPromptEncoder


class MusicTransformer(nn.Module):
    """
    Conditional Music Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        max_relative_position: int = 128,
        prompt_config: dict = None,
        # New optimization params
        num_kv_heads: int = 4,          # GQA: 8 heads query, 4 KV heads (2:1 ratio)
        use_qk_norm: bool = True,
        weight_tying: bool = True,
        ffn_ratio: float = 4.0,         # Can tune (3.5 ~ 4.0 recommended)
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.use_qk_norm = use_qk_norm
        self.weight_tying = weight_tying

        prompt_cfg = prompt_config or {}

        # Structured PromptEncoder (lightweight for training)
        self.structured_encoder = PromptEncoder(
            d_model=d_model,
            num_moods=prompt_cfg.get("num_moods", 10),
            num_genres=prompt_cfg.get("num_genres", 10),
            num_scenes=prompt_cfg.get("num_scenes", 10),
            num_tempos=prompt_cfg.get("num_tempos", 5),
            num_instruments=prompt_cfg.get("num_instruments", 8),
            num_energies=prompt_cfg.get("num_energies", 5),
            mood_dim=prompt_cfg.get("mood_dim", 64),
            genre_dim=prompt_cfg.get("genre_dim", 64),
            scene_dim=prompt_cfg.get("scene_dim", 64),
            tempo_dim=prompt_cfg.get("tempo_dim", 32),
            instrument_dim=prompt_cfg.get("instrument_dim", 32),
            energy_dim=prompt_cfg.get("energy_dim", 32),
        )

        # Lazy NLP encoder
        self._nlp_encoder = None
        self._nlp_d_model = d_model

        # Token Embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        d_ff = int(d_model * ffn_ratio)

        # Decoder blocks with GQA + QK-Norm
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                d_cond=d_model,
                dropout=dropout,
                num_kv_heads=num_kv_heads,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Weight Tying (strong regularization for small models)
        if weight_tying:
            self.output_projection.weight = self.token_embedding.embedding.weight

        self.dropout = nn.Dropout(dropout)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MusicTransformer] Parameters: {n_params:,} ({n_params/1e6:.1f}M) [GQA={num_kv_heads}, QK-Norm={use_qk_norm}, WeightTying={weight_tying}]")

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        prompts: List[str] = None,
        mood: torch.Tensor = None,
        genre: torch.Tensor = None,
        scene: torch.Tensor = None,
        tempo: torch.Tensor = None,
        instrument: torch.Tensor = None,
        energy: torch.Tensor = None,
        cond: torch.Tensor = None,  # pre-encoded conditioning (for optimized inference)
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with dual conditioning support (optimized).

        Supports:
        - NLP mode: pass `prompts: List[str]`
        - Structured mode: pass the 6 categorical tensors
        - Pre-encoded `cond` (for fast repeated inference)
        """
        B, T = tokens.shape
        device = tokens.device

        # --- Dual conditioning logic (core optimization) ---
        cond = self._get_conditioning(prompts, mood, genre, scene, tempo, instrument, energy, cond, device)

        # Token embedding
        x = self.token_embedding(tokens)
        x = self.dropout(x)

        new_kv_caches = []
        for i, block in enumerate(self.decoder_blocks):
            # Lấy cache của layer hiện tại nếu có
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cond, mask=None, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        # Output projection
        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits, new_kv_caches

    @property
    def nlp_encoder(self):
        """Lazy-loaded NLP encoder to avoid loading BERT during training."""
        if self._nlp_encoder is None:
            self._nlp_encoder = NLPPromptEncoder(d_model=self._nlp_d_model)
        return self._nlp_encoder

    def encode_prompt(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """Encode prompt only (for inference conditioning caching)."""
        return self.nlp_encoder(prompts, device=device)

    def encode_structured_prompt(
        self,
        mood: torch.Tensor,
        genre: torch.Tensor,
        scene: torch.Tensor,
        tempo: torch.Tensor,
        instrument: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """Fast structured prompt encoding (for training)."""
        return self.structured_encoder(mood, genre, scene, tempo, instrument, energy)

    def _get_conditioning(
        self,
        prompts: List[str] = None,
        mood: torch.Tensor = None,
        genre: torch.Tensor = None,
        scene: torch.Tensor = None,
        tempo: torch.Tensor = None,
        instrument: torch.Tensor = None,
        energy: torch.Tensor = None,
        cond: torch.Tensor = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Internal helper for clean dual + pre-encoded conditioning."""
        if cond is not None:
            return cond
        if prompts is not None and len(prompts) > 0 and isinstance(prompts[0], str):
            return self.nlp_encoder(prompts, device=device)
        if mood is not None:
            return self.structured_encoder(mood, genre, scene, tempo, instrument, energy)
        raise ValueError(
            "Must provide either `prompts` (List[str] for NLP), "
            "structured IDs (mood/genre/...), or pre-encoded `cond` tensor."
        )

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "vocab_size": self.vocab_size,
                    "d_model": self.d_model,
                    "max_seq_len": self.max_seq_len,
                },
            },
            path,
        )
        print(f"[MusicTransformer] Saved to {path}")

    @classmethod
    def load(cls, path: str, **override_kwargs) -> "MusicTransformer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        config.update(override_kwargs)

        model = cls(**config)

        # Basic vocab consistency check
        saved_vocab = config.get("vocab_size")
        if saved_vocab and saved_vocab != model.vocab_size:
            print(f"[WARNING] Vocab size mismatch: checkpoint={saved_vocab}, model={model.vocab_size}")

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[MusicTransformer] Loaded from {path}")
        return model
