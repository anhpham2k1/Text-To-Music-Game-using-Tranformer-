"""
Music Generation entry point.

Usage:
    python generate.py --mood happy --genre fantasy --scene village --tempo fast --instrument piano
    python generate.py --checkpoint checkpoints/best_model.pt --output outputs/my_music
"""

import os
import sys
import argparse
import yaml

import torch

from src.data.tokenizer import MidiTokenizer
from src.model.transformer import MusicTransformer
from src.inference.generator import MusicGenerator
from src.inference.renderer import MidiRenderer


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate Music from Text Prompt")

    # Prompt (text or structured)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Natural language prompt (recommended)")
    parser.add_argument("--mood", type=str, default=None)
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--tempo", type=str, default=None)
    parser.add_argument("--instrument", type=str, default=None)
    parser.add_argument("--energy", type=str, default=None)

    # Model
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="config/config.yaml")

    # Generation
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)

    # Output
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="background_music")
    parser.add_argument("--soundfont", type=str, default=None)

    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--piano_roll", action="store_true", help="Generate piano roll image")

    args = parser.parse_args()

    # Config
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    tok_cfg = config.get("tokenizer", {})
    audio_cfg = config.get("audio", {})

    print("=" * 60)
    print("  TEXT-TO-MUSIC: Generate")
    print("=" * 60)

    # --- Prompt ---
    if args.prompt:
        prompt = args.prompt
        structured = {}
    else:
        prompt = None
        structured = {
            "mood": args.mood, "genre": args.genre, "scene": args.scene,
            "tempo": args.tempo, "instrument": args.instrument, "energy": args.energy
        }
        print(f"\n  Using structured prompt: {structured}")

    print(f"\n  Prompt:")
    print(f"    {prompt or structured}")

    # --- Tokenizer ---
    tokenizer = MidiTokenizer(
        pitch_range=tuple(tok_cfg.get("pitch_range", [21, 108])),
        velocity_bins=tok_cfg.get("velocity_bins", 32),
        time_shift_bins=tok_cfg.get("time_shift_bins", 100),
    )

    # --- Load Model ---
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        print("  Train a model first: python train.py")
        print("  Or download a pre-trained model.")
        sys.exit(1)

    print(f"\n  Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {})

    loaded_vocab = model_config.get("vocab_size", tokenizer.vocab_size)
    model = MusicTransformer(
        vocab_size=loaded_vocab,
        d_model=model_config.get("d_model", model_cfg.get("d_model", 256)),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=model_config.get("max_seq_len", model_cfg.get("max_seq_len", 4096)),
        dropout=0.0,
        prompt_config=config.get("prompt", {}),
        num_kv_heads=4,
        use_qk_norm=True,
        weight_tying=True,
    )

    if loaded_vocab != tokenizer.vocab_size:
        print(f"[WARNING] Vocab mismatch: checkpoint={loaded_vocab} vs tokenizer={tokenizer.vocab_size}")

    model.load_state_dict(checkpoint["model_state_dict"])

    # --- Generator ---
    generator = MusicGenerator(model, tokenizer, device=args.device)

    # --- Generate ---
    print(f"\n  Generating music...")
    print(f"    Temperature: {args.temperature}")
    print(f"    Top-p: {args.top_p}")
    print(f"    Max length: {args.max_length} tokens")

    os.makedirs(args.output, exist_ok=True)
    midi_path = os.path.join(args.output, f"{args.name}.mid")

    generator.generate_midi(
        prompt=prompt,
        output_path=midi_path,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        **structured
    )

    # --- Render WAV ---
    wav_path = os.path.join(args.output, f"{args.name}.wav")
    soundfont = args.soundfont or audio_cfg.get("soundfont", "soundfonts/FluidR3_GM.sf2")

    renderer = MidiRenderer(
        soundfont_path=soundfont,
        sample_rate=audio_cfg.get("sample_rate", 44100),
    )
    renderer.render(midi_path, wav_path)

    # --- Piano Roll ---
    if args.piano_roll:
        try:
            from src.utils.visualization import plot_piano_roll
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
            piano_roll_path = os.path.join(args.output, f"{args.name}_piano_roll.png")
            plot_piano_roll(midi, output_path=piano_roll_path)
        except Exception as e:
            print(f"[WARNING] Could not generate piano roll: {e}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Generation complete!")
    print(f"     MIDI: {midi_path}")
    print(f"     WAV:  {wav_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
