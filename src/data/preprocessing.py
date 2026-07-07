"""
Data preprocessing utilities.

- Download dataset links
- Clean/filter MIDI files
- Generate auto-labels
"""

import os
import json
import shutil
from typing import Optional

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

from .tokenizer import MidiTokenizer


def filter_midi_files(
    input_dir: str,
    output_dir: str,
    min_duration: float = 5.0,
    max_duration: float = 300.0,
    min_notes: int = 10,
    max_notes: int = 50000,
    verbose: bool = True,
):
    """
    Lọc MIDI files theo tiêu chí chất lượng.

    Args:
        input_dir: Thư mục chứa MIDI gốc
        output_dir: Thư mục output (MIDI đã lọc)
        min_duration: Thời lượng tối thiểu (giây)
        max_duration: Thời lượng tối đa (giây)
        min_notes: Số nốt tối thiểu
        max_notes: Số nốt tối đa
    """
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    accepted = 0
    rejected = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith((".mid", ".midi")):
                continue

            total += 1
            filepath = os.path.join(root, filename)

            try:
                midi = pretty_midi.PrettyMIDI(filepath)
                duration = midi.get_end_time()
                num_notes = sum(len(inst.notes) for inst in midi.instruments)

                # Apply filters
                if duration < min_duration or duration > max_duration:
                    rejected += 1
                    continue
                if num_notes < min_notes or num_notes > max_notes:
                    rejected += 1
                    continue

                # Copy to output
                dest = os.path.join(output_dir, filename)
                # Handle duplicate filenames
                if os.path.exists(dest):
                    name, ext = os.path.splitext(filename)
                    dest = os.path.join(output_dir, f"{name}_{total}{ext}")

                shutil.copy2(filepath, dest)
                accepted += 1

            except Exception as e:
                rejected += 1
                if verbose:
                    print(f"  [SKIP] {filename}: {e}")

    print(f"\n[Filter] Total: {total} | Accepted: {accepted} | Rejected: {rejected}")
    return accepted


def generate_labels(
    midi_dir: str,
    output_path: str,
    tokenizer: Optional[MidiTokenizer] = None,
):
    """
    Sinh auto-labels cho tất cả MIDI files trong thư mục.

    Output: JSON file {filename: {mood, genre, scene, tempo, instrument, energy}}
    """
    if tokenizer is None:
        tokenizer = MidiTokenizer()

    labels = {}
    count = 0

    for root, _, files in os.walk(midi_dir):
        for filename in files:
            if not filename.lower().endswith((".mid", ".midi")):
                continue

            filepath = os.path.join(root, filename)
            try:
                label = tokenizer.auto_label(filepath)
                labels[filename] = label
                count += 1
            except Exception as e:
                print(f"  [SKIP] {filename}: {e}")
                labels[filename] = tokenizer._default_labels()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"[Labels] Generated labels for {count} files -> {output_path}")
    return labels


def print_dataset_info():
    """In thông tin các dataset có thể download. (Cập nhật 2026)"""
    info = """
╔══════════════════════════════════════════════════════════════════════╗
║                    RECOMMENDED DATASETS FOR GAME MUSIC AI            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. MAESTRO v3 (Piano - High Quality, giữ lại cho reference)         ║
║     URL: https://magenta.tensorflow.org/datasets/maestro              ║
║                                                                      ║
║  2. Lakh MIDI Dataset (Lớn, đa thể loại - KHUYẾN NGHỊ)               ║
║     URL: https://colinraffel.com/projects/lmd/                        ║
║     Download: LMD-full (176k files) hoặc clean subset                 ║
║     HF mirror: Kaggle "lakh-midi-clean"                               ║
║                                                                      ║
║  3. GigaMIDI (LỚN NHẤT - 1.4M+ files, đa dạng)                       ║
║     HF: https://huggingface.co/datasets/Metacreation/GigaMIDI         ║
║     Rất tốt để lọc theo instrument/genre                              ║
║                                                                      ║
║  4. MidiCaps (TỐT NHẤT cho Text-to-Music)                            ║
║     168k MIDI + text captions (mood, genre, instrument...)            ║
║     HF: https://huggingface.co/datasets/amaai-lab/MidiCaps            ║
║                                                                      ║
║  5. Video Game Music (VGMusic) - PHÙ HỢP NHẤT cho game background    ║
║     URL: https://www.vgmusic.com/ (~31k+ game MIDI files)             ║
║     Tải thủ công theo hệ máy hoặc thể loại                            ║
║                                                                      ║
║  6. Tegridy / LAKH MuseNet / ComMU / SymphonyNet (các collection)     ║
║     Tìm trên GitHub "Tegridy MIDI Dataset" hoặc HF                    ║
║                                                                      ║
║  USAGE (cập nhật):                                                   ║
║  1. Tải MIDI vào data/raw/ (có thể tạo subfolder: game_midi, lakh)   ║
║  2. python -m src.data.preprocessing                                 ║
║  3. Hoặc thủ công:                                                   ║
║     python -c "from src.data.preprocessing import filter_midi_files, generate_labels; filter_midi_files('data/raw', 'data/processed'); generate_labels('data/processed', 'data/labels/labels.json')" 
║                                                                      ║
║  LƯU Ý: Sau khi thêm data mới, chạy lại filter + generate_labels.    ║
║  Dùng MidiCaps hoặc VGMusic để tăng đa dạng cho game music.          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(info)


if __name__ == "__main__":
    print_dataset_info()
