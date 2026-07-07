"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional


class MusicRequest(BaseModel):
    """Request body for music generation.
    Supports either:
    - Free-form `prompt` (natural language)
    - Or structured game attributes (from frontend)
    """
    # Free text prompt (preferred for direct use)
    prompt: Optional[str] = Field(default=None, description="Miêu tả bản nhạc bằng ngôn ngữ tự nhiên")

    # Structured attributes (used by web UI)
    mood: Optional[str] = Field(default=None)
    genre: Optional[str] = Field(default=None)
    scene: Optional[str] = Field(default=None)
    tempo: Optional[str] = Field(default=None)
    instrument: Optional[str] = Field(default=None)
    energy: Optional[str] = Field(default=None)

    temperature: float = Field(default=0.85, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    max_length: int = Field(default=2048, ge=100, le=4096, description="Max tokens")


class MusicResponse(BaseModel):
    """Response body for music generation."""
    request_id: str
    midi_url: str
    wav_url: str
    duration: float
    num_notes: int
    prompt_text: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
