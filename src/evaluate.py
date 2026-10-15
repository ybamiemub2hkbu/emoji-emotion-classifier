"""Summarize the training runs into tables and figures.

Reads ``reports/runs.csv`` and ``reports/predictions.csv`` (written by ``src.train``)
and the saved models, then writes:

* ``reports/summary.md``: the result tables used in the README
* ``reports/final_holdout.csv``: the demo model's predictions on the holdout design
* ``reports/figures/*.png``: every figure in the README

Usage: ``python -m src.evaluate``
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import EXTERNAL_DIR, FIGURES_DIR, HOLDOUT_DIR, MODELS_DIR, REPORTS_DIR
from src.data import DESIGNS, load_external, load_frames
from src.gradcam import gradcam, overlay
from src.train import FINAL_MODEL, PREDICTIONS_CSV, RUNS_CSV

tf.get_logger().setLevel("ERROR")

MODEL_LABELS = {
    "baseline_cnn": "Classic CNN",
    "compact_cnn": "Compact CNN + augmentation",
    "mobilenet_v2": "MobileNetV2 (frozen, ImageNet)",
}
SHORT_LABELS = {"baseline_cnn": "Classic CNN", "compact_cnn": "Compact CNN", "mobilenet_v2": "MobileNetV2"}
PROTOCOL_LABELS = {
    "train": "Training\nframes",
    "random_split": "Random\nframe split",
    "leave_clips_out": "Unseen\nanimation clips",
    "other_designs": "Other emoji designs\n(trained on frames)",
    "leave_design_out": "Unseen design\n(+4 designs in training)",
}

# Chart palette: categorical slots 1-3 (validated colorblind-safe as a set),
# recessive chrome, and the status pair for correct / wrong.
COLORS = {"baseline_cnn": "#2a78d6", "compact_cnn": "#eb6834", "mobilenet_v2": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)
    print(f"  wrote reports/figures/{name}")


def _percent_axis(ax, axis="y"):
    formatter = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


# Segoe UI and Arial have no check-mark glyphs; matplotlib's bundled font does.
MARK_FONT = "DejaVu Sans"


def _pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def cluster_bootstrap_ci(correct, clusters, n_boot=5000, seed=0):
    """95% CI for accuracy, resampling whole clusters (e.g. all designs of one emoji)."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"correct": correct, "cluster": clusters})
    grouped = frame.groupby("cluster")["correct"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    picks = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))
    accs = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return np.percentile(accs, [2.5, 97.5])


def image_accuracy(predictions, external, protocol, model):
    """Accuracy of each emoji image averaged over seeds, with its codepoint and design."""
    p = predictions[(predictions["model"] == model) & (predictions["protocol"] == protocol)]
    p = p.merge(external[["image_id", "codepoint", "design"]], on="image_id")
    p = p.assign(correct=(p["p_happy"] >= 0.5) == p["label"])
    return p.groupby(["image_id", "codepoint", "design"], as_index=False)["correct"].mean()


def summarize(runs, predictions, external):
    """Return (headline table, per-fold table, {protocol: per-design table})."""
    rows = []
    for model in MODEL_LABELS:
        r = runs[runs["model"] == model]
        full = r[r["protocol"] == "other_designs"]
        rand = r[r["protocol"] == "random_split"]
        lco = r[r["protocol"] == "leave_clips_out"]
        fold_means = lco.groupby("fold")["test_accuracy"].mean()
        row = {
            "model": model,
            "train_acc": full["train_accuracy"].mean(),
            "random_acc": rand["test_accuracy"].mean(),
            "random_min": rand["test_accuracy"].min(),
            "random_max": rand["test_accuracy"].max(),
            "lco_acc": lco["test_accuracy"].mean(),
            "lco_fold_min": fold_means.min(),
            "lco_fold_max": fold_means.max(),
            "epochs_full": full["epochs"].mean(),
            "seconds_full": full["seconds"].mean(),
        }
        # Emoji test sets: CI from resampling the 16 emoji (all designs of an emoji together).
        for protocol, key in [("other_designs", "designs"), ("leave_design_out", "ldo")]:
            per_image = image_accuracy(predictions, external, protocol, model)
            if per_image.empty:
                continue
            low, high = cluster_bootstrap_ci(per_image["correct"], per_image["codepoint"])
            row.update({f"{key}_acc": per_image["correct"].mean(), f"{key}_ci_low": low, f"{key}_ci_high": high})
        rows.append(row)
    headline = pd.DataFrame(rows)

    folds = (
        runs[runs["protocol"] == "leave_clips_out"]
        .groupby(["fold", "model"])["test_accuracy"]
        .mean()
        .unstack("model")[list(MODEL_LABELS)]
    )

    designs = {}
    for protocol in ("other_designs", "leave_design_out"):
        per_image = pd.concat(
            [image_accuracy(predictions, external, protocol, m).assign(model=m) for m in MODEL_LABELS]
        )
        if not per_image.empty:
            table = per_image.groupby(["design", "model"])["correct"].mean().unstack("model")
            designs[protocol] = table.loc[list(DESIGNS), list(MODEL_LABELS)]
    return headline, folds, designs


def evaluate_holdout(model_name=FINAL_MODEL):
    """Score the demo model, trained on every labeled image, on the never-used sixth design."""
    path = MODELS_DIR / f"{model_name}_all_designs.keras"
    model = tf.keras.models.load_model(path)
    holdout, images = load_external(HOLDOUT_DIR)
    probs = model.predict(images, verbose=0)[:, 0]
    holdout = holdout.assign(p_happy=np.round(probs, 5), correct=(probs >= 0.5) == holdout["label"])
    holdout.drop(columns="path").to_csv(REPORTS_DIR / "final_holdout.csv", index=False)
    return holdout, images


def write_summary(headline, folds, designs, runs, holdout):
    def ci(r, key):
        if pd.isna(r.get(f"{key}_acc")):
            return "not run"
        return f"{_pct(r[f'{key}_acc'])} ({_pct(r[f'{key}_ci_low'])}–{_pct(r[f'{key}_ci_high'])})"

    lines = ["# Results", ""]
    lines += [
        "| Model | Training frames | Random frame split | Unseen clips | Other designs, trained on frames (95% CI) "
        "| Unseen design, +4 designs in training (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in headline.iterrows():
        lines.append(
            f"| {MODEL_LABELS[r['model']]} | {_pct(r['train_acc'])} | {_pct(r['random_acc'])} | "
            f"{_pct(r['lco_acc'])} | {ci(r, 'designs')} | {ci(r, 'ldo')} |"
        )
    lines += ["", "## Unseen clips, by held-out pair", ""]
    lines += ["| Held-out clips | " + " | ".join(SHORT_LABELS[m] for m in folds.columns) + " |"]
    lines += ["|---" * (len(folds.columns) + 1) + "|"]
    for fold, r in folds.iterrows():
        lines.append(f"| {fold.replace('test=', '')} | " + " | ".join(_pct(v) for v in r) + " |")
    titles = {
        "other_designs": "Other emoji designs, trained on the frames only",
        "leave_design_out": "Each design held out, trained on the frames + the other four designs",
    }
    for protocol, table in designs.items():
        lines += ["", f"## {titles[protocol]}", ""]
        lines += ["| Design | " + " | ".join(SHORT_LABELS[m] for m in table.columns) + " |"]
        lines += ["|---" * (len(table.columns) + 1) + "|"]
        for design, r in table.iterrows():
            lines.append(f"| {design} | " + " | ".join(_pct(v) for v in r) + " |")
    lines += ["", "## Training cost (all 80 frames)", ""]
    lines += ["| Model | Epochs | Seconds |", "|---|---|---|"]
    for _, r in headline.iterrows():
        lines.append(f"| {MODEL_LABELS[r['model']]} | {r['epochs_full']:.1f} | {r['seconds_full']:.0f} |")
    if holdout is not None:
        wrong = holdout.loc[~holdout["correct"], "name"].str.lower().tolist()
        lines += [
            "",
            f"## Final holdout ({holdout['design'].iloc[0]}, never used in any experiment)",
            "",
            f"Demo model ({SHORT_LABELS[FINAL_MODEL]} trained on all 160 labeled images): "
            f"{int(holdout['correct'].sum())}/{len(holdout)} correct ({_pct(holdout['correct'].mean())}).",
            "Wrong: " + (", ".join(wrong) if wrong else "none") + ".",
        ]
    seeds = runs.groupby("protocol")["seed"].nunique()
    lines += ["", f"Runs: {len(runs)} (seeds per configuration: {', '.join(f'{k} {v}' for k, v in seeds.items())})."]
    (REPORTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote reports/summary.md")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_frames(index, frames):
    """The four animations, seven frames each."""
    picks = [0, 3, 6, 9, 12, 15, 19]
    clips = ["happy1", "happy2", "sad1", "sad2"]
    fig, axes = plt.subplots(len(clips), len(picks), figsize=(9.2, 5.6))
    for row, clip in enumerate(clips):
        for col, frame in enumerate(picks):
            ax = axes[row, col]
            i = index.index[(index["clip"] == clip) & (index["frame"] == frame)][0]
            ax.imshow(frames[i].astype("uint8"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"frame {frame}", fontsize=9, color=INK_2, fontweight="normal")
            if col == 0:
                ax.set_ylabel(clip, fontsize=10, color=INK, rotation=0, ha="right", va="center")
    fig.suptitle(
        "All 80 training images are frames of four animations of one emoji",
        x=0.02, ha="left", fontsize=12, fontweight="semibold",
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.06, top=0.9)
    _save(fig, "training_frames.png")


def plot_protocols(headline):
    """Accuracy of each model under each evaluation protocol."""
    keys = {
        "train": ("train_acc", None, None),
        "random_split": ("random_acc", "random_min", "random_max"),
        "leave_clips_out": ("lco_acc", "lco_fold_min", "lco_fold_max"),
        "other_designs": ("designs_acc", "designs_ci_low", "designs_ci_high"),
        "leave_design_out": ("ldo_acc", "ldo_ci_low", "ldo_ci_high"),
    }
    protocols = [p for p in keys if keys[p][0] in headline]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width, gap = 0.085, 0.05
    x = np.arange(len(protocols))
    for k, model in enumerate(MODEL_LABELS):
        r = headline.set_index("model").loc[model]
        offset = (k - 1) * (width + gap)
        for j, protocol in enumerate(protocols):
            mean_key, lo_key, hi_key = keys[protocol]
            value = r[mean_key]
            ax.bar(x[j] + offset, value, width, color=COLORS[model], label=MODEL_LABELS[model] if j == 0 else None)
            if lo_key:
                ax.plot([x[j] + offset] * 2, [r[lo_key], r[hi_key]], color=INK_2, linewidth=1)
            top = r[hi_key] if hi_key else value
            ax.text(x[j] + offset, top + 0.015, f"{100 * value:.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK_2)
    ax.axhline(0.5, color=MUTED, linewidth=1)
    ax.text(len(protocols) - 0.5, 0.5, " chance", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, [PROTOCOL_LABELS[p] for p in protocols], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.5, len(protocols) - 0.5)
    _percent_axis(ax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9)
    ax.set_title("Accuracy depends on what the model is tested on", loc="left")
    _save(fig, "accuracy_by_protocol.png")


def plot_expression_curves(index, frames, predictions):
    """P(happy) along each animation when that animation was held out."""
    p = predictions[predictions["protocol"] == "leave_clips_out"].merge(
        index[["image_id", "clip", "frame"]], on="image_id"
    )
    curves = p.groupby(["model", "clip", "frame"])["p_happy"].mean()
    clips = ["happy1", "happy2", "sad1", "sad2"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.4), sharey=True)
    for ax, clip in zip(axes, clips):
        for model in MODEL_LABELS:
            series = curves.loc[(model, clip)]
            ax.plot(series.index, series.values, color=COLORS[model], linewidth=2, label=SHORT_LABELS[model])
        ax.axhline(0.5, color=MUTED, linewidth=1)
        truth = "happy" if clip.startswith("happy") else "sad"
        ax.text(0.5, 1.12, f"{clip}\ntrue label: {truth}", transform=ax.transAxes, ha="center",
                va="center", fontsize=9.5, color=INK)
        ax.set_xlabel("frame")
        ax.set_xticks([0, 5, 10, 15, 19])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        # First and last frame as thumbnails above the panel.
        for frame, xpos in [(0, 0.0), (19, 0.76)]:
            i = index.index[(index["clip"] == clip) & (index["frame"] == frame)][0]
            inset = ax.inset_axes([xpos, 1.02, 0.24, 0.24])
            inset.imshow(frames[i].astype("uint8"))
            inset.axis("off")
    axes[0].set_ylabel("P(happy), averaged over runs")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.suptitle(
        "Predicted P(happy) on animations the model never saw (first and last frame shown above each panel)",
        x=0.01, y=0.99, ha="left", fontsize=12, fontweight="semibold",
    )
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.07, right=0.99, wspace=0.1)
    _save(fig, "unseen_clip_curves.png")


def plot_more_designs(designs):
    """Per design: accuracy trained on frames only vs. with the other four designs added."""
    if "leave_design_out" not in designs:
        return
    before, after = designs["other_designs"], designs["leave_design_out"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    y = np.arange(len(before))[::-1]
    for ax, model in zip(axes, MODEL_LABELS):
        color = COLORS[model]
        for yi, b, a in zip(y, before[model], after[model]):
            ax.plot([b, a], [yi, yi], color=color, linewidth=2, alpha=0.5, zorder=2)
        ax.scatter(before[model], y, s=60, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=3)
        ax.scatter(after[model], y, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        mean_b, mean_a = before[model].mean(), after[model].mean()
        ax.set_title(f"{SHORT_LABELS[model]}: {100 * mean_b:.0f}% → {100 * mean_a:.0f}%", fontsize=10)
        ax.axvline(0.5, color=MUTED, linewidth=1, zorder=1)
        ax.set_xlim(0.2, 1.04)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        _percent_axis(ax, "x")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, before.index)
    keys = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE,
                   markeredgecolor=INK_2, markeredgewidth=2, label="trained on the frames only"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=INK_2,
                   label="trained on the frames + the other four designs"),
    ]
    fig.legend(handles=keys, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Accuracy on an emoji design the model never trained on",
        x=0.01, ha="left", fontsize=12, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, "more_designs.png")


def plot_separation(predictions):
    """Every P(happy) each model gave to the other-design emoji, split by true label."""
    p = predictions[predictions["protocol"] == "other_designs"]
    p = p.groupby(["model", "image_id", "label"], as_index=False)["p_happy"].mean()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    positions, ticks = [], []
    for k, model in enumerate(MODEL_LABELS):
        for j, (label, name) in enumerate([(0, "sad"), (1, "happy")]):
            pos = k * 3 + j
            values = p[(p["model"] == model) & (p["label"] == label)]["p_happy"].to_numpy()
            jitter = rng.uniform(-0.28, 0.28, len(values))
            ax.scatter(
                pos + jitter, values, s=26, color=COLORS[model], alpha=0.85,
                edgecolors=SURFACE, linewidths=1.2, zorder=3,
            )
            ax.plot([pos - 0.36, pos + 0.36], [np.median(values)] * 2, color=INK, linewidth=2, zorder=4)
            positions.append(pos)
            ticks.append(f"{name}\nemoji")
        ax.text(k * 3 + 0.5, 1.1, MODEL_LABELS[model], ha="center", fontsize=9.5, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=1)
    ax.set_xticks(positions, ticks, fontsize=8.5)
    ax.set_ylim(-0.04, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _percent_axis(ax)
    ax.set_ylabel("P(happy)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title(
        "Trained on the frames only: P(happy) for each other-design emoji (dot = image, bar = median)",
        loc="left", pad=22,
    )
    _save(fig, "other_designs_separation.png")


def plot_gallery(items, images, probs, title, filename):
    """Emoji grid (one row per design) with P(happy) under each image.

    ``items`` is a manifest (design, name, label, image_id); ``probs`` maps image_id
    to P(happy).
    """
    designs = list(dict.fromkeys(items["design"]))
    n_cols = items["name"].nunique()
    height = len(designs) * 1.25 + 0.45
    fig, axes = plt.subplots(len(designs), n_cols, figsize=(n_cols * 0.95, height), squeeze=False)
    for row, design in enumerate(designs):
        subset = items[items["design"] == design]
        for col, (i, item) in enumerate(subset.iterrows()):
            ax = axes[row, col]
            ax.imshow(images[i].astype("uint8"))
            ax.set_xticks([])
            ax.set_yticks([])
            prob = probs[item["image_id"]]
            correct = (prob >= 0.5) == item["label"]
            for spine in ax.spines.values():
                spine.set_visible(not correct)
                spine.set_color(CRITICAL)
                spine.set_linewidth(2)
            mark = "✓" if correct else "✗"
            ax.set_xlabel(f"{mark} {100 * prob:.0f}%", fontsize=7.5, color=GOOD if correct else CRITICAL,
                          labelpad=2, fontfamily=MARK_FONT)
            if col == 0:
                ax.set_ylabel(design, fontsize=9, color=INK, rotation=0, ha="right", va="center")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold", fontfamily=["Segoe UI", MARK_FONT])
    fig.subplots_adjust(wspace=0.08, hspace=0.42, top=1 - 0.4 / height)
    _save(fig, filename)


def plot_gradcam(index, frames, external, external_images):
    """Grad-CAM heatmaps from the seed-0 model of each architecture (trained on frames only)."""
    models = {name: tf.keras.models.load_model(MODELS_DIR / f"{name}.keras") for name in MODEL_LABELS}
    picks = [("frame", "happy2-10"), ("frame", "sad2-10")]
    picks += [("external", f"{design}/{name}") for design, name in [
        ("noto", "grinning-face-with-big-eyes"),
        ("twemoji", "slightly-smiling-face"),
        ("openmoji", "frowning-face"),
        ("fluent-3d", "crying-face"),
    ]]
    fig, axes = plt.subplots(len(picks), 1 + len(models), figsize=(2.1 * (1 + len(models)), 2.25 * len(picks)))
    for row, (kind, image_id) in enumerate(picks):
        if kind == "frame":
            i = index.index[index["image_id"] == image_id][0]
            image, label, title = frames[i], index.loc[i, "label"], f"training frame {image_id}"
        else:
            i = external.index[external["image_id"] == image_id][0]
            image, label = external_images[i], external.loc[i, "label"]
            title = f"{external.loc[i, 'design']}: {external.loc[i, 'name'].lower()}"
        axes[row, 0].imshow(image.astype("uint8"))
        axes[row, 0].set_title(title, fontsize=8.5, fontweight="normal", color=INK)
        for col, (name, model) in enumerate(models.items(), start=1):
            prob, cam = gradcam(model, image)
            axes[row, col].imshow(overlay(image, cam))
            correct = (prob >= 0.5) == label
            mark = "✓" if correct else "✗"
            axes[row, col].set_title(
                f"{mark} P(happy) {100 * prob:.0f}%", fontsize=8.5, fontweight="normal",
                color=GOOD if correct else CRITICAL, fontfamily=MARK_FONT,
            )
            if row == 0:
                axes[row, col].text(
                    0.5, 1.28, SHORT_LABELS[name], transform=axes[row, col].transAxes,
                    ha="center", fontsize=10, fontweight="semibold",
                )
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Where each model looks (Grad-CAM, brighter = more influence)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.subplots_adjust(wspace=0.06, hspace=0.32, top=0.91)
    _save(fig, "gradcam.png")


def main():
    runs = pd.read_csv(RUNS_CSV)
    predictions = pd.read_csv(PREDICTIONS_CSV)
    index, frames = load_frames()
    external, external_images = load_external(EXTERNAL_DIR)

    headline, folds, designs = summarize(runs, predictions, external)
    holdout = holdout_images = None
    if (MODELS_DIR / f"{FINAL_MODEL}_all_designs.keras").exists():
        holdout, holdout_images = evaluate_holdout()
    write_summary(headline, folds, designs, runs, holdout)
    print((REPORTS_DIR / "summary.md").read_text(encoding="utf-8"))

    plot_frames(index, frames)
    plot_protocols(headline)
    plot_expression_curves(index, frames, predictions)
    plot_more_designs(designs)
    plot_separation(predictions)

    def mean_probs(model, protocol):
        p = predictions[(predictions["model"] == model) & (predictions["protocol"] == protocol)]
        return p.groupby("image_id")["p_happy"].mean()

    plot_gallery(
        external, external_images, mean_probs("baseline_cnn", "other_designs"),
        "Classic CNN trained on the frames: P(happy) per emoji (left 8 happy, right 8 sad; ✗ = wrong)",
        "gallery_baseline_cnn.png",
    )
    if "leave_design_out" in designs:
        plot_gallery(
            external, external_images, mean_probs(FINAL_MODEL, "leave_design_out"),
            f"{SHORT_LABELS[FINAL_MODEL]}, each design held out in turn: P(happy) per emoji (✗ = wrong)",
            f"gallery_{FINAL_MODEL}_leave_design_out.png",
        )
    if holdout is not None:
        plot_gallery(
            holdout, holdout_images, holdout.set_index("image_id")["p_happy"],
            "Demo model on the final holdout design (left 8 happy, right 8 sad; ✗ = wrong)",
            "final_holdout.png",
        )
    plot_gradcam(index, frames, external, external_images)


if __name__ == "__main__":
    main()
