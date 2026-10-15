import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.config import IMG_SIZE
from src.data import (
    DESIGNS,
    HAPPY_EMOJI,
    SAD_EMOJI,
    frame_index,
    leave_clips_out_folds,
    random_frame_split,
    to_array,
)


def synthetic_index():
    rows = [
        {"image_id": f"{clip}-{frame:02d}", "clip": clip, "frame": frame, "label": int(clip.startswith("happy"))}
        for clip in ("happy1", "happy2", "sad1", "sad2")
        for frame in range(20)
    ]
    return pd.DataFrame(rows)


def test_transparent_pixels_become_white():
    # The real training PNGs store black under a fully transparent background.
    image = Image.new("RGBA", (150, 150), (0, 0, 0, 0))
    array = to_array(image)
    assert array.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert array.dtype == np.float32
    assert np.all(array == 255)


def test_reframe_crops_and_centers_the_content():
    image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    image.paste((200, 30, 30, 255), (10, 10, 90, 90))  # small square in a corner
    array = to_array(image, reframe=True)
    assert array.shape == (IMG_SIZE, IMG_SIZE, 3)
    np.testing.assert_allclose(array[IMG_SIZE // 2, IMG_SIZE // 2], [200, 30, 30], atol=2)
    np.testing.assert_allclose(array[0, 0], [255, 255, 255])


def test_opaque_images_are_resized_without_cropping():
    image = Image.new("RGB", (300, 300), (10, 120, 10))
    array = to_array(image, reframe=True)
    assert array.shape == (IMG_SIZE, IMG_SIZE, 3)
    np.testing.assert_allclose(array[IMG_SIZE // 2, IMG_SIZE // 2], [10, 120, 10], atol=2)


def test_frame_index_parses_file_names(tmp_path):
    for name in ["happy/happy1-00.png", "happy/happy2-13.png", "sad/sad1-07.png", "sad/sad2-19.png"]:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        Image.new("RGBA", (8, 8)).save(path)
    index = frame_index(tmp_path).set_index("image_id")
    assert list(index.loc[["happy1-00", "happy2-13", "sad1-07", "sad2-19"], "label"]) == [1, 1, 0, 0]
    assert index.loc["happy2-13", "clip"] == "happy2"
    assert index.loc["sad2-19", "frame"] == 19


def test_frame_index_rejects_unexpected_files(tmp_path):
    (tmp_path / "happy").mkdir()
    Image.new("RGBA", (8, 8)).save(tmp_path / "happy" / "smile.png")
    with pytest.raises(ValueError):
        frame_index(tmp_path)


def test_leave_clips_out_never_shares_a_clip():
    index = synthetic_index()
    folds = leave_clips_out_folds(index)
    assert len(folds) == 4
    held_out = []
    for _, train, test in folds:
        assert np.all(train ^ test)
        train_clips = set(index.loc[train, "clip"])
        test_clips = set(index.loc[test, "clip"])
        assert not train_clips & test_clips
        assert sorted(index.loc[test, "label"].unique()) == [0, 1]
        assert sorted(index.loc[train, "label"].unique()) == [0, 1]
        held_out += sorted(test_clips)
    # Every clip is held out in exactly two of the four folds.
    assert pd.Series(held_out).value_counts().eq(2).all()


def test_random_split_is_stratified_and_disjoint():
    index = synthetic_index()
    train, test = random_frame_split(index, test_size=0.25, seed=3)
    assert not np.any(train & test)
    assert test.sum() == 20
    assert index.loc[test, "label"].sum() == 10


def test_external_test_set_is_balanced():
    assert len(HAPPY_EMOJI) == len(SAD_EMOJI) == 8
    assert not set(HAPPY_EMOJI) & set(SAD_EMOJI)
    for url_for in DESIGNS.values():
        for codepoint, name in {**HAPPY_EMOJI, **SAD_EMOJI}.items():
            url = url_for(codepoint, name)
            assert url.startswith("https://") and url.endswith(".png")
            assert " " not in url
