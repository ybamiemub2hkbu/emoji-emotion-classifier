"""Download the image sets and turn them into arrays.

Three image sets are used:

* **Frames**: the 80-image happy-or-sad dataset. Every image is a frame of one of
  four short animations of the same 3D emoji (``happy1``, ``happy2``, ``sad1``,
  ``sad2``, 20 frames each).
* **Other designs**: 16 standard Unicode faces (8 happy, 8 sad) as drawn by five
  emoji designs. They test whether a model trained on the frames learned the
  expression or just this one emoji. In the leave-one-design-out experiment four
  designs are added to training and the fifth is the test set.
* **Final holdout**: the same 16 emoji from a sixth design (EmojiOne 2.2). No
  experiment uses it; it tests the demo model once, at the very end.

Run ``python -m src.data`` to download everything (about 15 MB).
"""

import hashlib
import http.client
import io
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    BACKGROUND,
    EXTERNAL_DIR,
    FRAMES_DIR,
    HOLDOUT_DIR,
    FRAMES_SHA256,
    FRAMES_URL,
    IMG_SIZE,
)

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

# In the training frames the face spans about 96% of the image width.
FACE_FILL = 0.96


def to_array(image, reframe=False):
    """Convert a PIL image to a float32 array of shape (IMG_SIZE, IMG_SIZE, 3) in [0, 255].

    Transparent pixels are composited onto ``BACKGROUND``. The training PNGs store
    black under their transparent background, so simply dropping the alpha channel
    (what ``tf.keras.utils.load_img`` does) would put every face on black.

    With ``reframe=True`` the image is cropped to its visible content and padded to a
    square so the face fills the frame the same way it does in the training frames.
    """
    image = image.convert("RGBA")

    if reframe:
        alpha = image.getchannel("A").point(lambda a: 255 if a > 128 else 0)
        bbox = alpha.getbbox()
        if bbox:
            image = image.crop(bbox)
        side = round(max(image.size) / FACE_FILL)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = canvas

    background = Image.new("RGBA", image.size, BACKGROUND + (255,))
    image = Image.alpha_composite(background, image).convert("RGB")
    if image.size != (IMG_SIZE, IMG_SIZE):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def load_image(path, reframe=False):
    with Image.open(path) as image:
        return to_array(image, reframe=reframe)


# ---------------------------------------------------------------------------
# Training frames
# ---------------------------------------------------------------------------

FRAME_PATTERN = re.compile(r"^(?P<clip>(?P<label>happy|sad)\d+)-(?P<frame>\d+)\.png$")


def _fetch(url, retries=5):
    request = urllib.request.Request(url, headers={"User-Agent": "emoji-emotion-classifier"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, http.client.HTTPException):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def download_frames(target=FRAMES_DIR):
    """Download and verify the happy-or-sad zip, then extract it to ``target``."""
    if target.exists() and len(list(target.glob("*/*.png"))) == 80:
        return target

    payload = _fetch(FRAMES_URL)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FRAMES_SHA256:
        raise RuntimeError(f"Checksum mismatch for {FRAMES_URL}: got {digest}")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            name = Path(member)
            if name.suffix != ".png" or ".." in name.parts or name.is_absolute():
                continue
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
    return target


def frame_index(frames_dir=FRAMES_DIR):
    """One row per training frame: path, clip, frame number and label (1 = happy)."""
    rows = []
    for path in sorted(frames_dir.glob("*/*.png")):
        match = FRAME_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"Unexpected file name: {path.name}")
        rows.append(
            {
                "image_id": path.stem,
                "path": str(path),
                "clip": match["clip"],
                "frame": int(match["frame"]),
                "label": int(match["label"] == "happy"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No frames in {frames_dir}. Run `python -m src.data` first.")
    return pd.DataFrame(rows)


def load_frames(frames_dir=FRAMES_DIR):
    index = frame_index(frames_dir)
    images = np.stack([load_image(path) for path in index["path"]])
    return index, images


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def leave_clips_out_folds(index):
    """The four ways to hold out one happy clip and one sad clip.

    Returns a list of ``(fold_name, train_mask, test_mask)``. The model trains on the
    other two clips, so every test frame comes from an animation it never saw.
    """
    happy_clips = sorted(index.loc[index["label"] == 1, "clip"].unique())
    sad_clips = sorted(index.loc[index["label"] == 0, "clip"].unique())
    folds = []
    for happy in happy_clips:
        for sad in sad_clips:
            test = index["clip"].isin([happy, sad]).to_numpy()
            folds.append((f"test={happy}+{sad}", ~test, test))
    return folds


def random_frame_split(index, test_size=0.25, seed=0):
    """A conventional stratified random split of frames, ignoring which clip they came from."""
    positions = np.arange(len(index))
    train, test = train_test_split(
        positions, test_size=test_size, stratify=index["label"], random_state=seed
    )
    train_mask = np.isin(positions, train)
    return train_mask, ~train_mask


# ---------------------------------------------------------------------------
# Other emoji designs (test only)
# ---------------------------------------------------------------------------

HAPPY_EMOJI = {
    "1f600": "Grinning face",
    "1f603": "Grinning face with big eyes",
    "1f604": "Grinning face with smiling eyes",
    "1f601": "Beaming face with smiling eyes",
    "1f606": "Grinning squinting face",
    "1f642": "Slightly smiling face",
    "1f60a": "Smiling face with smiling eyes",
    "263a": "Smiling face",
}

SAD_EMOJI = {
    "1f641": "Slightly frowning face",
    "2639": "Frowning face",
    "1f61e": "Disappointed face",
    "1f614": "Pensive face",
    "1f61f": "Worried face",
    "1f622": "Crying face",
    "1f62d": "Loudly crying face",
    "1f625": "Sad but relieved face",
}

# Pinned commits so the test set never changes underneath the results.
_FLUENT = "https://raw.githubusercontent.com/microsoft/fluentui-emoji/1ffb34c752ecf5d402f04cfb4b392c77f57c54bc"
_NOTO = "https://raw.githubusercontent.com/googlefonts/noto-emoji/e20cbc2bbec1926686be9f9bee7d1d2cfa1fea0e"
_TWEMOJI = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0"
_OPENMOJI = "https://raw.githubusercontent.com/hfg-gmuend/openmoji/aeb8bb3a59e2de39c754ac79180c8131c906acea"
_EMOJIONE = "https://cdn.jsdelivr.net/gh/joypixels/emojione@v2.2.7"


def _fluent_url(codepoint, name):
    slug = name.lower().replace(" ", "_")
    return f"{_FLUENT}/assets/{urllib.parse.quote(name)}/3D/{slug}_3d.png"


# Ordered roughly from most to least similar to the training emoji.
DESIGNS = {
    "Fluent 3D": _fluent_url,
    "Noto 3D": lambda cp, name: f"{_NOTO}/3D/png/512/emoji_u{cp}.png",
    "Noto": lambda cp, name: f"{_NOTO}/2D/png/512/emoji_u{cp}.png",
    "Twemoji": lambda cp, name: f"{_TWEMOJI}/assets/72x72/{cp}.png",
    "OpenMoji": lambda cp, name: f"{_OPENMOJI}/color/618x618/{cp.upper()}.png",
}

# A sixth design kept out of every experiment. It is used once, at the very end, to
# test the demo model that was trained on everything above.
HOLDOUT_DESIGNS = {
    "EmojiOne 2.2": lambda cp, name: f"{_EMOJIONE}/assets/png/{cp}.png",
}


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def download_external(target=EXTERNAL_DIR, designs=None):
    """Download every emoji in every design and write a manifest."""
    designs = DESIGNS if designs is None else designs
    rows = []
    emoji = [(cp, name, 1) for cp, name in HAPPY_EMOJI.items()]
    emoji += [(cp, name, 0) for cp, name in SAD_EMOJI.items()]
    for design, url_for in designs.items():
        for codepoint, name, label in emoji:
            path = target / _slug(design) / f"{_slug(name)}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_fetch(url_for(codepoint, name)))
            rows.append(
                {
                    "image_id": f"{_slug(design)}/{_slug(name)}",
                    "path": str(path.relative_to(target)),
                    "design": design,
                    "codepoint": codepoint,
                    "name": name,
                    "label": label,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(target / "manifest.csv", index=False)
    return manifest


def load_external(target=EXTERNAL_DIR):
    if not (target / "manifest.csv").exists():
        raise FileNotFoundError(f"No manifest in {target}. Run `python -m src.data` first.")
    manifest = pd.read_csv(target / "manifest.csv", dtype={"codepoint": str})
    images = np.stack([load_image(target / path, reframe=True) for path in manifest["path"]])
    return manifest, images


if __name__ == "__main__":
    download_frames()
    index = frame_index()
    print(f"Frames: {len(index)} images, clips {sorted(index['clip'].unique())}")
    for title, target, designs in [
        ("Other designs", EXTERNAL_DIR, DESIGNS),
        ("Final holdout", HOLDOUT_DIR, HOLDOUT_DESIGNS),
    ]:
        manifest = download_external(target, designs)
        counts = manifest.groupby("design")["label"].agg(["count", "sum"])
        print(f"{title}: {len(manifest)} images")
        for design, row in counts.iterrows():
            print(f"  {design:12s} {row['count']} images ({row['sum']} happy)")
