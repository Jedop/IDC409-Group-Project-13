"""
Project-wide settings: folder paths, dataset options, ASR model names and thresholds.

Planned contents:
- ROOT / DATA_DIR / AUDIO folders / OUTPUT_DIR / DB_PATH
- LibriSpeech download URL and subset size
- noise levels (SNR in dB) for the clear-vs-noisy extension
- ASR model names to compare
- thresholds (low-confidence flag, "hard clip" WER threshold)
- ensure_dirs() helper
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AUDIO_CLEAN = DATA_DIR / "audio" / "clean"
AUDIO_NOISY = DATA_DIR / "audio" / "noisy"
SAMPLE_DIR = DATA_DIR / "sample_audio"
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
DB_PATH = DATA_DIR / "speech_project.db"
MANIFEST_CSV = DATA_DIR / "manifest.csv"

# ---- dataset -------------------------------------------------------------
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
NUM_CLIPS = 60            # number of clean clips to keep
MAX_PER_SPEAKER = 10      # spread clips over several speakers
MAX_DURATION_S = 15.0     # keep clips short so CPU transcription stays fast
SAMPLE_RATE = 16000
SNR_LEVELS_DB = [10, 5, 0]  # white-noise versions of every clean clip
NUM_SAMPLE_CLIPS = 5        # small sample committed to GitHub

# ---- ASR -----------------------------------------------------------------
ASR_MODELS = ["tiny.en", "base.en"]   # two pretrained systems to compare
LOW_CONFIDENCE = 0.50                 # below this -> note "low confidence"

# ---- ML ------------------------------------------------------------------
RANDOM_SEED = 42
HARD_WER_THRESHOLD = 0.15   # clip is "hard" if its WER is above this
CLEAN_SNR_PROXY_DB = 40.0   # stand-in SNR value for clean / unknown-noise clips
CORR_THRESHOLD = 0.90       # drop one of two features correlated above this


def ensure_dirs():
    for p in (AUDIO_CLEAN, AUDIO_NOISY, SAMPLE_DIR, OUTPUT_DIR, PLOTS_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)
