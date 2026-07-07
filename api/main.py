"""
FastAPI backend for Text-to-Music Generator.

Endpoints:
    POST /generate     — Generate music from text prompt
    GET  /download/{id}/{format} — Download generated MIDI/WAV
    GET  /health       — Health check

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import yaml

import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import MusicRequest, MusicResponse, HealthResponse
from src.data.tokenizer import MidiTokenizer
from src.model.transformer import MusicTransformer
from src.inference.generator import MusicGenerator
from src.inference.renderer import MidiRenderer


# Globals
generator = None
renderer = None
model_device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan (replaces deprecated @app.on_event("startup"))."""
    global generator, renderer, model_device

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml"
    )
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    tok_cfg = config.get("tokenizer", {})
    audio_cfg = config.get("audio", {})

    # Tokenizer
    tokenizer = MidiTokenizer(
        pitch_range=tuple(tok_cfg.get("pitch_range", [21, 108])),
        velocity_bins=tok_cfg.get("velocity_bins", 32),
        time_shift_bins=tok_cfg.get("time_shift_bins", 100),
    )

    # Model
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "best_model.pt",
    )

    if os.path.exists(checkpoint_path):
        print(f"[API] Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mc = checkpoint.get("config", {})

        loaded_vocab = mc.get("vocab_size", tokenizer.vocab_size)
        model = MusicTransformer(
            vocab_size=loaded_vocab,
            d_model=mc.get("d_model", model_cfg.get("d_model", 256)),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 6),
            d_ff=model_cfg.get("d_ff", 1024),
            max_seq_len=mc.get("max_seq_len", 2048),
            dropout=0.0,
            prompt_config=config.get("prompt", {}),
            num_kv_heads=4,
            use_qk_norm=True,
            weight_tying=True,
        )
        if loaded_vocab != tokenizer.vocab_size:
            print(f"[API WARNING] Vocab mismatch: checkpoint={loaded_vocab} vs tokenizer={tokenizer.vocab_size}")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"[API] No checkpoint found at {checkpoint_path}")
        print("[API] Using untrained model (random weights) for demo")
        model = MusicTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=model_cfg.get("d_model", 256),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 6),
            d_ff=model_cfg.get("d_ff", 1024),
            max_seq_len=model_cfg.get("max_seq_len", 2048),
            dropout=0.0,
            prompt_config=config.get("prompt", {}),
            num_kv_heads=4,
            use_qk_norm=True,
            weight_tying=True,
        )

    device = "auto"
    generator = MusicGenerator(model, tokenizer, device=device)
    model_device = str(generator.device)

    # Renderer
    sf_path = audio_cfg.get("soundfont", "soundfonts/FluidR3_GM.sf2")
    renderer = MidiRenderer(
        soundfont_path=sf_path,
        sample_rate=audio_cfg.get("sample_rate", 44100),
    )

    os.makedirs("outputs", exist_ok=True)
    print(f"[API] Ready! Device: {model_device}")

    yield

    # Shutdown cleanup
    print("[API] Shutting down...")
    generator = None
    renderer = None


# ============================================================
# App setup (with modern lifespan)
# ============================================================

app = FastAPI(
    title="Text-to-Music Generator",
    description="Sinh nhạc nền cho game từ mô tả văn bản",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ============================================================
# Endpoints
# ============================================================

@app.get("/", include_in_schema=False)
async def root():
    """Serve frontend."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Text-to-Music API", "docs": "/docs"}


def _build_prompt_from_structured(req: MusicRequest) -> str:
    """Convert structured game attributes into natural language prompt."""
    mood = req.mood or "peaceful"
    genre = req.genre or "fantasy"
    scene = req.scene or "village"
    tempo = req.tempo or "moderate"
    instrument = req.instrument or "piano"
    energy = req.energy or "medium"

    return (
        f"A {mood} {genre} {scene} background music, "
        f"{tempo} tempo, {instrument} as main instrument, {energy} energy."
    )

@app.post("/generate", response_model=MusicResponse)
async def generate_music(request: MusicRequest):
    """Generate music from text prompt or structured attributes."""
    if generator is None:
        raise HTTPException(503, "Model not loaded")

    request_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join("outputs", request_id)
    os.makedirs(output_dir, exist_ok=True)

    # Support both free text and structured (from frontend)
    if request.prompt:
        prompt = request.prompt
    else:
        prompt = _build_prompt_from_structured(request)

    # Generate MIDI
    midi_path = os.path.join(output_dir, "background_music.mid")
    generator.generate_midi(
        prompt=prompt,
        output_path=midi_path,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    # Render WAV
    wav_path = os.path.join(output_dir, "background_music.wav")
    renderer.render(midi_path, wav_path)

    # Get info
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
        duration = midi.get_end_time()
        num_notes = sum(len(inst.notes) for inst in midi.instruments)
    except Exception:
        duration = 0.0
        num_notes = 0

    return MusicResponse(
        request_id=request_id,
        midi_url=f"/download/{request_id}/midi",
        wav_url=f"/download/{request_id}/wav",
        duration=round(duration, 1),
        num_notes=num_notes,
        prompt_text=prompt,
    )


@app.get("/download/{request_id}/{format}")
async def download(request_id: str, format: str):
    """Download generated MIDI or WAV file."""
    if format == "midi":
        ext = "mid"
        media_type = "audio/midi"
    elif format == "wav":
        ext = "wav"
        media_type = "audio/wav"
    else:
        raise HTTPException(400, f"Invalid format: {format}. Use 'midi' or 'wav'.")

    path = os.path.join("outputs", request_id, f"background_music.{ext}")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found. Generate music first.")

    return FileResponse(
        path,
        media_type=media_type,
        filename=f"background_music.{ext}",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        model_loaded=generator is not None,
        device=model_device,
    )
