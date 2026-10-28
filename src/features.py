"""
Feature engineering, feature understanding and feature reduction.

Planned contents:
- build_feature_table(): one row per clip x ASR model with features that do NOT need the true transcript
  (duration, SNR, predicted word count, speaking rate, confidence, model) and the targets (wer, hard clip)
- drop_correlated(): remove one feature of each highly correlated pair
- analyse_features(): correlation matrix, PCA explained variance, random-forest importance
- main(): save outputs/features.csv, feature_analysis.txt and plots
Run with: python -m src.features
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from . import config, database, metrics
from .text_utils import tokenize, normalize_text

BASE_FEATURES = ["duration_s", "snr_db_filled", "n_pred_words", "pred_chars",
                 "avg_pred_word_len", "pred_speaking_rate", "confidence"]


def build_feature_table():
    df = database.read_joined()
    if df.empty:
        raise SystemExit("No transcripts found - run python -m src.transcribe first.")
    df = metrics.add_wer(df)
    pred = df["predicted_text"].fillna("")
    df["n_pred_words"] = pred.map(lambda t: len(tokenize(t)))
    df["pred_chars"] = pred.map(lambda t: len(normalize_text(t)))
    df["avg_pred_word_len"] = df["pred_chars"] / df["n_pred_words"].clip(lower=1)
    df["pred_speaking_rate"] = df["n_pred_words"] / df["duration_s"]
    df["snr_db_filled"] = df["snr_db"].fillna(config.CLEAN_SNR_PROXY_DB)
    df["confidence"] = df["confidence"].fillna(0.0)
    dummies = pd.get_dummies(df["asr_model"], prefix="model", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df["hard"] = (df["wer"] > config.HARD_WER_THRESHOLD).astype(int)
    df = df.dropna(subset=["wer"]).reset_index(drop=True)
    return df, BASE_FEATURES + list(dummies.columns)


def drop_correlated(X: pd.DataFrame, threshold=config.CORR_THRESHOLD):
    X = X.loc[:, X.var() > 0]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    kept = [c for c in X.columns if c not in dropped]
    return kept, dropped


def analyse_features(df, cols):
    X, y = df[cols], df["wer"]
    kept, dropped = drop_correlated(X)
    scaled = StandardScaler().fit_transform(X.loc[:, X.var() > 0])
    pca = PCA().fit(scaled)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    rf = RandomForestRegressor(n_estimators=200, random_state=config.RANDOM_SEED).fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    return {"corr": X.corr(), "kept": kept, "dropped": dropped,
            "cum_var": cum, "n95": n95, "importance": importance}


def main():
    config.ensure_dirs()
    df, cols = build_feature_table()
    df.to_csv(config.OUTPUT_DIR / "features.csv", index=False)
    a = analyse_features(df, cols)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(a["corr"].values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im); ax.set_title("Feature correlation")
    plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "features_correlation.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(a["cum_var"]) + 1), a["cum_var"], marker="o")
    plt.axhline(0.95, color="grey", ls="--"); plt.xlabel("PCA components")
    plt.ylabel("Cumulative explained variance"); plt.title("PCA")
    plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "features_pca_variance.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    a["importance"].sort_values().plot.barh(color="#4C72B0")
    plt.xlabel("Random-forest importance (predicting WER)"); plt.title("Feature importance")
    plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "features_importance.png", dpi=130); plt.close()

    text = ["=== FEATURE ANALYSIS ===", f"Features ({len(cols)}): {', '.join(cols)}", "",
            f"Dropped for correlation > {config.CORR_THRESHOLD}: {a['dropped'] or 'none'}",
            f"Kept ({len(a['kept'])}): {', '.join(a['kept'])}", "",
            f"PCA: {a['n95']} components explain 95% of the variance (of {len(cols)}).", "",
            "Random-forest importance:", a["importance"].round(3).to_string()]
    (config.OUTPUT_DIR / "feature_analysis.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
