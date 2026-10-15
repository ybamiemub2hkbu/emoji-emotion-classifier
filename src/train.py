"""Train every model under every evaluation protocol.

Protocols
---------
random_split     75/25 stratified random split of the frames (the usual approach).
leave_clips_out  Hold out one happy and one sad animation (4 folds), so every test
                 frame comes from a clip the model never saw.
other_designs    Train on all 80 frames, test on the same 16 emoji drawn by five
                 other emoji designs. Training accuracy from these runs is what the
                 standard "train until 99.9%" setup reports.
leave_design_out Train on the frames plus four of the five designs, test on the fifth
                 (5 folds, 3 seeds). Measures what more varied training data buys.

Each (model, protocol, fold, seed) writes one row to ``reports/runs.csv`` and one row
per test image to ``reports/predictions.csv``. Finished runs are skipped, so an
interrupted job picks up where it stopped. The seed-0 model trained on all frames is
saved to ``models/<name>.keras``.

Usage::

    python -m src.train                                # everything (about 80 min on a laptop CPU)
    python -m src.train --models compact_cnn --seeds 0 --protocols leave_clips_out
    python -m src.train --final                        # demo model trained on every labeled image
"""

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from src.config import MODELS_DIR, REPORTS_DIR, SEEDS
from src.data import leave_clips_out_folds, load_external, load_frames, random_frame_split
from src.models import BUILDERS

BATCH_SIZE = 10

# Fixed before looking at any test result: the baseline uses the classic recipe
# (stop as soon as training accuracy reaches 99.9%, at most 15 epochs); the other two
# get enough epochs for their training loss to level off under augmentation.
RECIPES = {
    "baseline_cnn": {"epochs": 15, "stop_at_accuracy": 0.999},
    "compact_cnn": {"epochs": 100},
    "mobilenet_v2": {"epochs": 20},
}

# Leave-one-design-out trains 15 models per seed on 144 images, so it uses fewer seeds.
LEAVE_DESIGN_OUT_SEEDS = (0, 1, 2)

# The architecture shipped as the demo model. It ties the classic CNN on
# leave-one-design-out accuracy, is far more robust when trained on the frames alone,
# looks at the mouth rather than the outline (Grad-CAM), and is an 8x smaller file.
FINAL_MODEL = "mobilenet_v2"

RUNS_CSV = REPORTS_DIR / "runs.csv"
PREDICTIONS_CSV = REPORTS_DIR / "predictions.csv"


class StopAtAccuracy(tf.keras.callbacks.Callback):
    """Stop training once the epoch's training accuracy reaches ``threshold``."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("accuracy", 0) >= self.threshold:
            self.model.stop_training = True


def fit(name, images, labels, seed):
    tf.keras.utils.set_random_seed(seed)
    model = BUILDERS[name]()
    recipe = RECIPES[name]
    dataset = (
        tf.data.Dataset.from_tensor_slices((images, labels.astype("float32")))
        .shuffle(len(images), seed=seed)
        .batch(BATCH_SIZE)
    )
    callbacks = []
    if "stop_at_accuracy" in recipe:
        callbacks.append(StopAtAccuracy(recipe["stop_at_accuracy"]))
    history = model.fit(
        dataset, epochs=recipe["epochs"], callbacks=callbacks, shuffle=False, verbose=0
    )
    return model, len(history.history["loss"])


def predict(model, images):
    return model.predict(images, batch_size=32, verbose=0)[:, 0]


def accuracy(labels, probs):
    return float(np.mean((probs >= 0.5) == labels))


def log_loss(labels, probs):
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


PROTOCOLS = ("random_split", "leave_clips_out", "other_designs", "leave_design_out")


def experiments(index, frames, external, external_images, seed, protocols=PROTOCOLS):
    """Yield (protocol, fold, train_x, train_y, test_x, test_y, test_ids) for one seed."""
    y = index["label"].to_numpy()
    ids = index["image_id"].to_numpy()
    ext_y = external["label"].to_numpy()
    ext_ids = external["image_id"].to_numpy()

    if "random_split" in protocols:
        train, test = random_frame_split(index, seed=seed)
        yield "random_split", "75/25", frames[train], y[train], frames[test], y[test], ids[test]
    if "leave_clips_out" in protocols:
        for fold, train, test in leave_clips_out_folds(index):
            yield "leave_clips_out", fold, frames[train], y[train], frames[test], y[test], ids[test]
    if "other_designs" in protocols:
        yield "other_designs", "all frames", frames, y, external_images, ext_y, ext_ids
    if "leave_design_out" in protocols and seed in LEAVE_DESIGN_OUT_SEEDS:
        for design in external["design"].unique():
            test = (external["design"] == design).to_numpy()
            yield (
                "leave_design_out",
                design,
                np.concatenate([frames, external_images[~test]]),
                np.concatenate([y, ext_y[~test]]),
                external_images[test],
                ext_y[test],
                ext_ids[test],
            )


def main(model_names, seeds, protocols=PROTOCOLS):
    tf.config.experimental.enable_op_determinism()
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    index, frames = load_frames()
    external, external_images = load_external()

    done = set()
    if RUNS_CSV.exists():
        previous = pd.read_csv(RUNS_CSV)
        done = set(zip(previous["model"], previous["protocol"], previous["fold"], previous["seed"]))

    for seed in seeds:
        for name in model_names:
            splits = experiments(index, frames, external, external_images, seed, protocols)
            for protocol, fold, train_x, train_y, test_x, test_y, test_ids in splits:
                if (name, protocol, fold, seed) in done:
                    continue
                start = time.time()
                model, epochs = fit(name, train_x, train_y, seed)
                probs = predict(model, test_x)
                run = {
                    "model": name,
                    "protocol": protocol,
                    "fold": fold,
                    "seed": seed,
                    "n_train": len(train_y),
                    "n_test": len(test_y),
                    "epochs": epochs,
                    "train_accuracy": accuracy(train_y, predict(model, train_x)),
                    "test_accuracy": accuracy(test_y, probs),
                    "test_log_loss": log_loss(test_y, probs),
                    "seconds": round(time.time() - start, 1),
                }
                predictions = pd.DataFrame(
                    {
                        "model": name,
                        "protocol": protocol,
                        "fold": fold,
                        "seed": seed,
                        "image_id": test_ids,
                        "label": test_y,
                        "p_happy": np.round(probs, 5),
                    }
                )
                pd.DataFrame([run]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False)
                predictions.to_csv(
                    PREDICTIONS_CSV, mode="a", header=not PREDICTIONS_CSV.exists(), index=False
                )
                if protocol == "other_designs" and seed == 0:
                    model.save(MODELS_DIR / f"{name}.keras")
                print(
                    f"seed {seed}  {name:13s} {protocol:16s} {fold:22s} "
                    f"train {run['train_accuracy']:.3f}  test {run['test_accuracy']:.3f}  "
                    f"({run['epochs']} epochs, {run['seconds']}s)",
                    flush=True,
                )
                tf.keras.backend.clear_session()


def train_final(name=FINAL_MODEL, seed=0):
    """Train on every labeled image (frames + all five designs) for the command-line demo."""
    tf.config.experimental.enable_op_determinism()
    index, frames = load_frames()
    external, external_images = load_external()
    images = np.concatenate([frames, external_images])
    labels = np.concatenate([index["label"].to_numpy(), external["label"].to_numpy()])
    model, epochs = fit(name, images, labels, seed)
    path = MODELS_DIR / f"{name}_all_designs.keras"
    model.save(path)
    print(f"Trained {name} on {len(labels)} images for {epochs} epochs -> {path.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=list(RECIPES), choices=list(RECIPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument(
        "--final", action="store_true", help="train the demo model on every labeled image and exit"
    )
    args = parser.parse_args()
    if args.final:
        train_final()
    else:
        main(args.models, args.seeds, args.protocols)
