"""
Prompt Encoder — chuyển structured game prompt thành embedding vectors.

Phương án A (chính): Attribute Embedding
- Mỗi attribute (mood, genre, scene, tempo, instrument, energy) có embedding riêng
- Concat → Project → prompt embedding

Phương án B (tùy chọn): Text-based (BERT)
- Dùng pre-trained BERT-tiny encode free-form text
"""

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    """
    Attribute-based Prompt Encoder.

    Mỗi thuộc tính game prompt (mood, genre, scene, tempo, instrument, energy)
    có embedding table riêng. Các embeddings được concat rồi project
    thành prompt vector cho cross-attention conditioning.

    Ưu điểm so với BERT:
    - Nhẹ (~200KB vs ~17MB)
    - Nhanh (single embedding lookup)
    - Dễ kiểm soát — chỉ hiểu các attributes đã định nghĩa
    """

    def __init__(
        self,
        d_model: int = 256,
        num_moods: int = 10,
        num_genres: int = 10,
        num_scenes: int = 10,
        num_tempos: int = 5,
        num_instruments: int = 8,
        num_energies: int = 5,
        mood_dim: int = 64,
        genre_dim: int = 64,
        scene_dim: int = 64,
        tempo_dim: int = 32,
        instrument_dim: int = 32,
        energy_dim: int = 32,
    ):
        super().__init__()

        # Embedding tables
        self.mood_emb = nn.Embedding(num_moods, mood_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_dim)
        self.scene_emb = nn.Embedding(num_scenes, scene_dim)
        self.tempo_emb = nn.Embedding(num_tempos, tempo_dim)
        self.instrument_emb = nn.Embedding(num_instruments, instrument_dim)
        self.energy_emb = nn.Embedding(num_energies, energy_dim)

        # Concat dimension
        concat_dim = mood_dim + genre_dim + scene_dim + tempo_dim + instrument_dim + energy_dim

        # Original robust concat + projection (reliable)
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.d_model = d_model

    def forward(
        self,
        mood: torch.Tensor,
        genre: torch.Tensor,
        scene: torch.Tensor,
        tempo: torch.Tensor,
        instrument: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        emb = torch.cat([
            self.mood_emb(mood),
            self.genre_emb(genre),
            self.scene_emb(scene),
            self.tempo_emb(tempo),
            self.instrument_emb(instrument),
            self.energy_emb(energy),
        ], dim=-1)
        prompt_emb = self.projection(emb)
        return prompt_emb.unsqueeze(1)

    @property
    def output_dim(self) -> int:
        return self.d_model


class NLPPromptEncoder(nn.Module):
    """
    Phác thảo kiến trúc Text-to-Music Prompt Encoder sử dụng Pre-trained LLM.
    Cho phép nhận vào văn bản tự nhiên (Natural Language) thay vì các thuộc tính rời rạc.
    """
    def __init__(self, d_model: int = 256, bert_model_name: str = "prajjwal1/bert-tiny"):
        super().__init__()
        try:
            import warnings
            from transformers import BertModel, BertTokenizer
            # Suppress expected "some weights not used" warning from bert-tiny
            warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
        except ImportError:
            raise ImportError("Vui lòng cài đặt thư viện transformers: pip install transformers")

        self.d_model = d_model
        
        # 1. Tải mô hình BERT (Đóng băng trọng số để train nhanh hơn)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Đóng băng BERT (Freeze)
        for param in self.bert.parameters():
            param.requires_grad = False

        # Dim của bert-tiny là 128, bert-base là 768
        bert_hidden_size = self.bert.config.hidden_size

        # 2. Lớp chiếu (Projection Layer) để map từ BERT dim sang Music Decoder dim
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            texts: List các câu miêu tả tự nhiên (VD: ["Bản nhạc piano buồn", "Nhạc dồn dập"])
            
        Returns:
            (batch_size, seq_len, d_model) — Vector cho Cross-Attention
        """
        # Tokenize text
        encoded = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors="pt"
        ).to(device)

        # Đưa qua BERT
        with torch.no_grad():
            outputs = self.bert(**encoded)
            
        # Lấy hidden states của tất cả các token (để Decoder có thể "attend" vào từng từ)
        # outputs.last_hidden_state có shape: (batch_size, text_seq_len, bert_hidden_size)
        text_embeddings = outputs.last_hidden_state

        # Chiếu sang kích thước của Music Transformer
        # shape: (batch_size, text_seq_len, d_model)
        prompt_emb = self.projection(text_embeddings)

        return prompt_emb
