import numpy as np
import pytest
import tensorflow as tf

from src.config import IMG_SIZE
from src.gradcam import gradcam
from src.models import baseline_cnn, compact_cnn, mobilenet_v2
from src.train import StopAtAccuracy

BUILDERS = {
    "baseline_cnn": baseline_cnn,
    "compact_cnn": compact_cnn,
    # No download in tests: random weights exercise the same graph.
    "mobilenet_v2": lambda: mobilenet_v2(weights=None),
}


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, size=(2, IMG_SIZE, IMG_SIZE, 3)).astype("float32")


@pytest.mark.parametrize("name", BUILDERS)
def test_models_output_one_probability_per_image(name, images):
    model = BUILDERS[name]()
    probs = model.predict(images, verbose=0)
    assert probs.shape == (2, 1)
    assert np.all((probs >= 0) & (probs <= 1))


def test_baseline_matches_the_classic_architecture():
    assert baseline_cnn().count_params() == 9_494_561


def test_compact_cnn_is_about_one_percent_of_the_baseline():
    assert compact_cnn().count_params() < 0.011 * baseline_cnn().count_params()


def test_mobilenet_trains_only_the_head():
    model = mobilenet_v2(weights=None)
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    assert trainable == 1280 + 1


@pytest.mark.parametrize("name", BUILDERS)
def test_gradcam_heatmap_matches_the_image(name, images):
    model = BUILDERS[name]()
    prob, heatmap = gradcam(model, images[0])
    assert heatmap.shape == (IMG_SIZE, IMG_SIZE)
    assert heatmap.min() >= 0 and heatmap.max() <= 1 + 1e-6
    expected = float(model.predict(images[:1], verbose=0)[0, 0])
    assert prob == pytest.approx(expected, abs=1e-4)


def test_stop_at_accuracy_callback():
    callback = StopAtAccuracy(0.999)
    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    model.stop_training = False  # normally set by fit()
    callback.set_model(model)
    callback.on_epoch_end(0, {"accuracy": 0.95})
    assert not callback.model.stop_training
    callback.on_epoch_end(1, {"accuracy": 1.0})
    assert callback.model.stop_training
