"""
ASR evaluation: word error rate (WER).

Planned contents:
- word_error_counts(): substitutions / deletions / insertions for one clip (jiwer)
- add_wer(): add wer + error-count columns to a joined DataFrame
- summarise_wer(): mean per-clip WER and corpus WER per ASR model and condition
- main(): write WER back to the database and save outputs/wer_per_clip.csv + wer_summary.csv
Run with: python -m src.metrics
"""
import jiwer
import numpy as np
import pandas as pd

from . import config, database
from .text_utils import normalize_text, condition_sort_key


def word_error_counts(ref, hyp):
    """Return (substitutions, deletions, insertions, n_ref_words)."""
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0, 0, len(hyp.split()), 0
    if not hyp:
        return 0, n_ref, 0, n_ref
    out = jiwer.process_words(ref, hyp)
    return out.substitutions, out.deletions, out.insertions, n_ref


def add_wer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts = [word_error_counts(r, h) for r, h in zip(df["true_text"], df["predicted_text"])]
    df[["n_sub", "n_del", "n_ins", "n_ref_words"]] = pd.DataFrame(counts, index=df.index)
    errors = df["n_sub"] + df["n_del"] + df["n_ins"]
    df["wer"] = np.where(df["n_ref_words"] > 0, errors / df["n_ref_words"].replace(0, np.nan), np.nan)
    return df


def summarise_wer(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["asr_model", "condition"])
    out = g.agg(n_clips=("clip_id", "count"),
                mean_wer=("wer", "mean"),
                median_wer=("wer", "median"),
                sub=("n_sub", "sum"), dele=("n_del", "sum"), ins=("n_ins", "sum"),
                ref_words=("n_ref_words", "sum")).reset_index()
    out["corpus_wer"] = (out["sub"] + out["dele"] + out["ins"]) / out["ref_words"]
    out["_k"] = out["condition"].map(condition_sort_key)
    out = out.sort_values(["asr_model", "_k"]).drop(columns="_k")
    return out.round(4)


def main():
    config.ensure_dirs()
    df = database.read_joined()
    if df.empty:
        raise SystemExit("No transcripts found - run python -m src.transcribe first.")
    df = add_wer(df)
    database.update_wer(zip(df["wer"], df["clip_id"], df["asr_model"]))
    df.to_csv(config.OUTPUT_DIR / "wer_per_clip.csv", index=False)
    summary = summarise_wer(df)
    summary.to_csv(config.OUTPUT_DIR / "wer_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
