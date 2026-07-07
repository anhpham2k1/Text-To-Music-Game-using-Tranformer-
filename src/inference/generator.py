"""
Music Generator — Autoregressive generation pipeline with KV Caching.

Sinh nhạc từ prompt bằng Music Transformer + sampling.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

from .sampling import combined_sampling


class MusicGenerator:
    """
    Generator for Music Transformer (with RoPE & KV Cache).

    Usage:
        generator = MusicGenerator(model, tokenizer, device)
        tokens = generator.generate("A happy piano tune", temperature=0.85, top_p=0.9)
    """

    def __init__(self, model, tokenizer, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str = None,
        max_length: int = 2048,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        # Structured prompt support (optional)
        mood: int = None,
        genre: int = None,
        scene: int = None,
        tempo: int = None,
        instrument: int = None,
        energy: int = None,
    ) -> List[int]:
        """
        Sinh nhạc autoregressive từ NLP prompt (O(N) with KV Cache).

        Optimized: pre-encode text conditioning once (avoids re-running BERT every step).
        """
        # --- Start with <BOS> token ---
        generated = [self.tokenizer.bos_id]
        kv_caches = None

        # Pre-encode conditioning once (supports both text and structured)
        if prompt is not None:
            cond = self.model.encode_prompt([prompt], device=self.device)
        else:
            # Structured path
            bs = 1
            mood_t = torch.tensor([mood or 3], device=self.device)
            genre_t = torch.tensor([genre or 0], device=self.device)
            scene_t = torch.tensor([scene or 2], device=self.device)
            tempo_t = torch.tensor([tempo or 2], device=self.device)
            inst_t = torch.tensor([instrument or 0], device=self.device)
            energy_t = torch.tensor([energy or 2], device=self.device)
            cond = self.model.encode_structured_prompt(mood_t, genre_t, scene_t, tempo_t, inst_t, energy_t)

        for step in range(max_length - 1):
            # Nếu có cache, ta chỉ cần truyền token mới nhất vào (sequence length = 1)
            last_token = generated[-1:]
            
            input_tensor = torch.tensor(
                [last_token], dtype=torch.long, device=self.device
            )

            # Forward pass with pre-encoded cond (fast)
            logits, kv_caches = self.model(
                tokens=input_tensor,
                cond=cond,
                kv_caches=kv_caches,
            )

            # Get logits for last position
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Sample next token
            next_token = combined_sampling(
                next_logits.clone(),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            token_id = next_token.item()
            generated.append(token_id)

            # Stop at <EOS>
            if token_id == self.tokenizer.eos_id:
                break

        return generated

    @torch.no_grad()
    def generate_midi(
        self,
        prompt: str = None,
        output_path: str = "output.mid",
        max_length: int = 2048,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        # Structured support
        mood: int = None, genre: int = None, scene: int = None,
        tempo: int = None, instrument: int = None, energy: int = None,
    ) -> str:
        """
        Sinh nhạc và lưu thành file MIDI.

        Args:
            prompt: Natural text prompt
            output_path: Đường dẫn file MIDI output
            max_length, temperature, top_p, top_k: Sampling params

        Returns:
            str — đường dẫn file MIDI đã lưu
        """
        import os

        # Generate tokens
        tokens = self.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            mood=mood, genre=genre, scene=scene,
            tempo=tempo, instrument=instrument, energy=energy,
        )

        # Decode tokens → MIDI
        # Lấy tempo mặc định là 120, model sẽ tự chọn thời gian qua time_shift tokens
        default_tempo = 120

        midi = self.tokenizer.decode(tokens, default_tempo=default_tempo)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        midi.write(output_path)

        duration = midi.get_end_time()
        num_notes = sum(len(inst.notes) for inst in midi.instruments)

        print(f"[Generator] Saved MIDI: {output_path}")
        print(f"  Duration: {duration:.1f}s | Notes: {num_notes} | Tokens: {len(tokens)}")

        return output_path
