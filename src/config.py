"""Paths and constants shared by every script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
FRAMES_DIR = DATA_DIR / "happy-or-sad"
EXTERNAL_DIR = DATA_DIR / "external"
HOLDOUT_DIR = DATA_DIR / "holdout"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Every image is converted to this size before it reaches a model.
IMG_SIZE = 150

# Transparent pixels are composited onto this color (RGB).
BACKGROUND = (255, 255, 255)

# Label convention used everywhere: 1 = happy, 0 = sad.
CLASS_NAMES = ("sad", "happy")

FRAMES_URL = "https://storage.googleapis.com/learning-datasets/happy-or-sad.zip"
FRAMES_SHA256 = "891cbe64fa5437598d9769baba2d68550dc3c688fb59523a42bfafc1a8f8cf6a"

SEEDS = (0, 1, 2, 3, 4)
