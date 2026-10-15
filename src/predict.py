"""Classify a face image as happy or sad.

Usage::

    python -m src.predict path/to/face.png
    python -m src.predict face.png --gradcam heatmap.png
    python -m src.predict face.png --model models/compact_cnn.keras

Transparent backgrounds are filled with white and the image is cropped and padded
to a square, the same preprocessing used for the test emoji.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES, MODELS_DIR
from src.data import load_image
from src.gradcam import gradcam, overlay
from src.train import FINAL_MODEL

# Trained on all 160 labeled images (frames + five emoji designs) by `python -m src.train --final`.
DEFAULT_MODEL = MODELS_DIR / f"{FINAL_MODEL}_all_designs.keras"


def classify(image_paths, model_path=DEFAULT_MODEL, gradcam_path=None):
    model = tf.keras.models.load_model(model_path)
    results = []
    for path in image_paths:
        image = load_image(path, reframe=True)
        if gradcam_path:
            prob, cam = gradcam(model, image)
            side_by_side = np.concatenate([image.astype("uint8"), overlay(image, cam)], axis=1)
            target = Path(gradcam_path)
            if len(image_paths) > 1:
                target = target.with_name(f"{target.stem}_{Path(path).stem}{target.suffix}")
            Image.fromarray(side_by_side).save(target)
        else:
            prob = float(model.predict(image[None], verbose=0)[0, 0])
        results.append((path, CLASS_NAMES[int(prob >= 0.5)], prob))
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify face images as happy or sad.")
    parser.add_argument("images", nargs="+", help="image files (PNG, JPG, ...)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to a saved .keras model")
    parser.add_argument("--gradcam", metavar="PNG", help="also save the image next to its Grad-CAM heatmap")
    args = parser.parse_args()

    for path, label, prob in classify(args.images, args.model, args.gradcam):
        print(f"{label:5s}  P(happy) = {prob:.3f}  {path}")


if __name__ == "__main__":
    main()
