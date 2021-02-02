import os
import click
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa


from dataloader import (
    BalancedMelSampler,
    MelSampler,
)
from losses import NpairsLoss
from metrics import TFLWLRAP
from models import NUM_FRAMES, Classifier, DeepMetricLearning
from split_data import get_split


os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def get_model(
    saved_path="",
    pretrained_with_contrastive=False,
    pretrained_path="",
):
    if pretrained_with_contrastive:
        model = DeepMetricLearning()
    else:
        model = Classifier()
        model._build()

        print(f" -> Loading weights from {pretrained_path}\n")
        model.load_weights(pretrained_path, by_name=True)

    os.makedirs(saved_path, exist_ok=True)
    return model


def get_callbacks(pretrained_with_contrastive=False, fold_id=0, saved_path=""):
    if pretrained_with_contrastive:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(saved_path, f"pretrained_best_fold{fold_id}.h5"),
                monitor="val_loss",
                save_weights_only=True,
                save_best_only=True,
                mode="min",
                save_freq="epoch",
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=30,
                monitor="val_loss",
                mode="min",
            ),
        ]

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            saved_path, f"fold{fold_id}", "model-{val_lwlrap:.3f}-{val_loss:.3f}.h5"
        ),
        monitor="val_lwlrap",
        save_weights_only=True,
        save_best_only=True,
        mode="max",
        save_freq="epoch",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=20, monitor="val_lwlrap", mode="max"
    )
    return [model_checkpoint, early_stopping]
