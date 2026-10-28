"""
Text analysis on the transcripts: keyword detection and matching to the expected answer.

Planned contents:
- select_keywords(): pick the most common content words in the true transcripts
- detect_keywords(): which keywords appear in a transcript
- keyword_metrics(): keyword precision / recall / F1 (true transcript = ground truth)
- match_to_expected(): similarity between predicted and true transcript, exact + fuzzy match accuracy
- main(): save outputs/keywords.txt, keyword_metrics.csv, keyword_per_word.csv,
  matching_results.csv, matching_summary.csv
Run with: python -m src.text_tasks
"""
import difflib
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config, database
from .text_utils import tokenize, normalize_text, condition_sort_key

MATCH_THRESHOLD = 0.85  # similarity needed for a "fuzzy match"


def select_keywords(true_texts, n=20):
    """Most frequent content words (document frequency), skipping stop words."""
    counts = Counter()
    for t in true_texts:
        counts.update({w for w in tokenize(t) if len(w) > 3 and w not in ENGLISH_STOP_WORDS})
    return [w for w, _ in counts.most_common(n)]


def detect_keywords(text, keywords):
    words = set(tokenize(text))
    return {k for k in keywords if k in words}


def keyword_metrics(df, keywords):
    """Precision/recall/F1 per (asr_model, condition) and per keyword."""
    recs = []
    for _, r in df.iterrows():
        truth, pred = detect_keywords(r["true_text"], keywords), detect_keywords(r["predicted_text"], keywords)
        for k in keywords:
            recs.append({"asr_model": r["asr_model"], "condition": r["condition"], "keyword": k,
                         "tp": int(k in truth and k in pred), "fp": int(k not in truth and k in pred),
                         "fn": int(k in truth and k not in pred)})
    recs = pd.DataFrame(recs)

    def prf(g):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return pd.Series({"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r, "f1": f})

    by_group = recs.groupby(["asr_model", "condition"]).apply(prf).reset_index()
    by_group["_k"] = by_group["condition"].map(condition_sort_key)
    by_group = by_group.sort_values(["asr_model", "_k"]).drop(columns="_k")
    by_word = recs.groupby(["asr_model", "keyword"]).apply(prf).reset_index()
    return by_group.round(4), by_word.round(4)


def match_to_expected(df, threshold=MATCH_THRESHOLD):
    out = df[["clip_id", "asr_model", "condition", "true_text", "predicted_text"]].copy()
    out["similarity"] = [difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
                         for a, b in zip(out["true_text"], out["predicted_text"])]
    out["exact_match"] = [normalize_text(a) == normalize_text(b)
                          for a, b in zip(out["true_text"], out["predicted_text"])]
    out["fuzzy_match"] = out["similarity"] >= threshold
    summary = out.groupby(["asr_model", "condition"]).agg(
        n_clips=("clip_id", "count"), exact_accuracy=("exact_match", "mean"),
        fuzzy_accuracy=("fuzzy_match", "mean"), mean_similarity=("similarity", "mean")).reset_index()
    summary["_k"] = summary["condition"].map(condition_sort_key)
    summary = summary.sort_values(["asr_model", "_k"]).drop(columns="_k")
    return out, summary.round(4)


def main():
    config.ensure_dirs()
    df = database.read_joined()
    if df.empty:
        raise SystemExit("No transcripts found - run python -m src.transcribe first.")
    df["predicted_text"] = df["predicted_text"].fillna("")
    clean_truth = df[df["condition"] == "clean"].drop_duplicates("clip_id")["true_text"]
    keywords = select_keywords(clean_truth if len(clean_truth) else df["true_text"])
    (config.OUTPUT_DIR / "keywords.txt").write_text("\n".join(keywords), encoding="utf-8")

    by_group, by_word = keyword_metrics(df, keywords)
    by_group.to_csv(config.OUTPUT_DIR / "keyword_metrics.csv", index=False)
    by_word.to_csv(config.OUTPUT_DIR / "keyword_per_word.csv", index=False)
    per_clip, summary = match_to_expected(df)
    per_clip.to_csv(config.OUTPUT_DIR / "matching_results.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / "matching_summary.csv", index=False)
    print("Keywords:", ", ".join(keywords))
    print("\nKeyword detection:\n", by_group.to_string(index=False))
    print("\nTranscript matching:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
