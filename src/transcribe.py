"""
Speech-to-text with a pretrained ASR model (faster-whisper, no training).

Planned contents:
- load_model(): load a Whisper model (tiny.en / base.en) on CPU
- transcribe_file(): return normalized text, confidence score and notes for one audio file
- run(): transcribe every clip not yet done, save to the database,
  then write outputs/transcripts_<model>.csv and outputs/transcripts_<model>.txt
Run with: python -m src.transcribe --model tiny.en
"""
import argparse

import numpy as np

from . import config, database
from .text_utils import normalize_text


def load_model(name):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe_file(model, path):
    segments, _ = model.transcribe(str(path), language="en", beam_size=5)
    segs = list(segments)
    text = normalize_text(" ".join(s.text.strip() for s in segs))
    if segs:
        weights = [max(s.end - s.start, 1e-3) for s in segs]
        confidence = float(np.exp(np.average([s.avg_logprob for s in segs], weights=weights)))
    else:
        confidence = 0.0
    notes = []
    if not text:
        notes.append("empty transcript")
    if confidence < config.LOW_CONFIDENCE:
        notes.append("low confidence")
    return text, confidence, "; ".join(notes)


def write_outputs(model_name):
    df = database.read_joined(model_name)
    cols = ["clip_id", "filename", "condition", "snr_db", "true_text",
            "predicted_text", "confidence", "notes"]
    safe = model_name.replace("/", "_")
    csv_path = config.OUTPUT_DIR / f"transcripts_{safe}.csv"
    txt_path = config.OUTPUT_DIR / f"transcripts_{safe}.txt"
    df[cols].to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"{r['filename']}\t{r['predicted_text']}\n")
    print(f"Saved {csv_path.name} and {txt_path.name} ({len(df)} clips)")


def run(model_name, limit=None, overwrite=False):
    import pandas as pd
    config.ensure_dirs()
    clips = database.read_clips()
    if clips.empty:
        raise SystemExit("No clips in the database - run python -m src.data_prep first.")
    if overwrite:
        database.delete_transcripts(model_name)
    clips = clips[~clips["clip_id"].isin(database.transcribed_ids(model_name))]
    if limit:
        clips = clips.head(limit)
    print(f"Transcribing {len(clips)} clips with {model_name} ...")
    model, batch = load_model(model_name), []
    for i, (_, r) in enumerate(clips.iterrows(), 1):
        text, conf, notes = transcribe_file(model, config.ROOT / r["filepath"])
        batch.append({"clip_id": r["clip_id"], "asr_model": model_name,
                      "predicted_text": text, "confidence": conf, "notes": notes})
        if i % 20 == 0 or i == len(clips):
            database.insert_transcripts(pd.DataFrame(batch))
            batch = []
            print(f"  {i}/{len(clips)} done")
    write_outputs(model_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.ASR_MODELS[0])
    ap.add_argument("--limit", type=int, help="only transcribe this many clips (quick test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run(a.model, a.limit, a.overwrite)
