"""The three architectures compared in this project.

Every model takes raw RGB images in [0, 255] of shape (150, 150, 3) and returns
P(happy). Scaling happens inside the model, so a saved model can be used on a
plain image array with no separate preprocessing step.

Each model ends in a ``logit`` Dense layer followed by a sigmoid, and names the
feature map that Grad-CAM should explain in ``GRADCAM_LAYER``.
"""

import tensorflow as tf

from src.config import IMG_SIZE

layers = tf.keras.layers

GRADCAM_LAYER = {
    "baseline_cnn": "last_conv",
    "compact_cnn": "last_conv",
    "mobilenet_v2": "backbone",
}


def augmentation():
    """Random geometry and color changes, active only during training.

    Inputs are still in [0, 255] here, so pixels exposed by a rotation, shift or
    zoom are filled with white to match the background.
    """
    fill = {"fill_mode": "constant", "fill_value": 255.0}
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05, **fill),
            layers.RandomZoom((-0.15, 0.25), **fill),
            layers.RandomTranslation(0.08, 0.08, **fill),
            layers.RandomContrast(0.3, value_range=(0, 255)),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="augmentation",
    )


def _head(x, inputs, name, dropout):
    if dropout:
        x = layers.Dropout(dropout)(x)
    logit = layers.Dense(1, name="logit")(x)
    prob = layers.Activation("sigmoid", name="prob")(logit)
    return tf.keras.Model(inputs, prob, name=name)


def baseline_cnn(augment=False):
    """Three convolution blocks and a 512-unit dense layer (9.5M parameters).

    This is the textbook architecture that reaches 99.9% training accuracy on the
    happy-or-sad frames within a few epochs. Nearly all of its parameters sit in
    the dense layer after ``Flatten``.
    """
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augmentation()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", name="last_conv")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    model = _head(x, inputs, "baseline_cnn", dropout=0)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compact_cnn(augment=True):
    """Four convolution blocks and global average pooling (98K parameters).

    Global average pooling replaces ``Flatten`` + ``Dense(512)``, which removes 99%
    of the baseline's parameters and stops the classifier from keying on exact
    pixel positions.
    """
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augmentation()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    model = _head(x, inputs, "compact_cnn", dropout=0.3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def mobilenet_v2(augment=True, weights="imagenet"):
    """A frozen ImageNet MobileNetV2 with a new single-unit head (1,281 trainable parameters)."""
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augmentation()(inputs) if augment else inputs
    x = layers.Resizing(160, 160)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), include_top=False, weights=weights, name="backbone"
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    model = _head(x, inputs, "mobilenet_v2", dropout=0.2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


BUILDERS = {
    "baseline_cnn": baseline_cnn,
    "compact_cnn": compact_cnn,
    "mobilenet_v2": mobilenet_v2,
}
