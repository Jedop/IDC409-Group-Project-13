"""
Extension: clear vs noisy audio, and comparison of the two ASR systems.

Planned contents:
- WER per condition (clean / noisy SNR levels / own accented recordings) for every ASR model
- line plot of WER against noise level
- paired comparison of the two ASR models (mean WER difference, clips where each wins, Wilcoxon test)
- save outputs/condition_comparison.csv, outputs/asr_comparison.csv and plots
Run with: python -m src.compare_conditions
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, database, metrics
from .text_utils import condition_sort_key


def main():
    config.ensure_dirs()
    joined = database.read_joined()
    if joined.empty:
        raise SystemExit("No transcripts found - run python -m src.transcribe first.")
    df = metrics.add_wer(joined)
    conds = sorted(df["condition"].unique(), key=condition_sort_key)
    models = sorted(df["asr_model"].unique())

    table = df.pivot_table(index="condition", columns="asr_model", values="wer", aggfunc="mean").loc[conds]
    if "clean" in table.index:
        for m in models:
            table[f"{m}_increase_vs_clean"] = table[m] - table.loc["clean", m]
    table.round(4).to_csv(config.OUTPUT_DIR / "condition_comparison.csv")
    print("Mean WER by condition:\n", table.round(3).to_string())

    plt.figure(figsize=(6.5, 4))
    for m in models:
        plt.plot(conds, table[m].values, marker="o", label=m)
    plt.ylabel("Mean WER"); plt.xlabel("Condition (left = clean, right = noisiest)")
    plt.title("Effect of noise on WER"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "compare_wer_vs_noise.png", dpi=130); plt.close()

    if len(models) >= 2:
        a, b = models[0], models[1]
        wide = df.pivot_table(index="clip_id", columns="asr_model", values="wer").dropna(subset=[a, b])
        diff = wide[a] - wide[b]
        row = {"model_a": a, "model_b": b, "n_clips": len(wide),
               "mean_wer_a": wide[a].mean(), "mean_wer_b": wide[b].mean(),
               "mean_diff_a_minus_b": diff.mean(),
               "a_better": int((diff < 0).sum()), "b_better": int((diff > 0).sum()),
               "ties": int((diff == 0).sum())}
        try:
            from scipy.stats import wilcoxon
            row["wilcoxon_p"] = float(wilcoxon(wide[a], wide[b]).pvalue)
        except ValueError:
            row["wilcoxon_p"] = np.nan
        pd.DataFrame([row]).round(4).to_csv(config.OUTPUT_DIR / "asr_comparison.csv", index=False)
        print("\nASR comparison:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
