"""Grad-CAM: which parts of the image pushed the model toward its answer.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.
"""

import matplotlib
import numpy as np
import tensorflow as tf

from src.models import GRADCAM_LAYER


def gradcam(model, image, layer_name=None):
    """Return ``(p_happy, heatmap)`` for one image of shape (H, W, 3) in [0, 255].

    The heatmap has the image's height and width, values in [0, 1], and highlights
    the regions that pushed the prediction toward the class the model chose.

    All three models are a straight chain of layers, so the forward pass is run one
    layer at a time to capture the feature map and the pre-sigmoid logit.
    """
    layer_name = layer_name or GRADCAM_LAYER[model.name]
    x = tf.convert_to_tensor(np.asarray(image, dtype="float32")[None])
    features = logit = None
    with tf.GradientTape() as tape:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x, training=False)
            if layer.name == layer_name:
                features = x
                tape.watch(features)
            if layer.name == "logit":
                logit = x[0, 0]
                break
        if features is None or logit is None:
            raise ValueError(f"{model.name} has no layer named {layer_name!r} or 'logit'")
        # Explain whichever class the model predicted.
        score = tf.where(logit >= 0, logit, -logit)

    grads = tape.gradient(score, features)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear")[..., 0]
    return float(tf.sigmoid(logit)), cam.numpy()


def overlay(image, heatmap, alpha=0.55, cmap="inferno"):
    """Blend a [0, 1] heatmap over a grayscale copy of an image in [0, 255]; returns uint8 RGB.

    The image is shown in gray because the hot end of the colormap is yellow, which
    would vanish against a yellow emoji.
    """
    gray = np.asarray(image, dtype="float32") @ np.array([0.299, 0.587, 0.114], dtype="float32")
    colors = matplotlib.colormaps[cmap](heatmap)[..., :3] * 255
    blended = (1 - alpha) * gray[..., None] + alpha * colors
    return np.clip(blended, 0, 255).astype("uint8")
