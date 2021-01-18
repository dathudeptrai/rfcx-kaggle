import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import random

import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import json
import logging

import click
import pandas as pd
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from tqdm import tqdm

from dataloader import (BalancedMelSampler, MelSampler,
                        convert_csv_to_dict_for_dataloader)
from losses import NpairsLoss, label_ranking_loss_tf
from metrics import TFLWLRAP, LwlrapAccumulator
from models import NUM_FRAMES, Classifier, DeepMetricLearning
from split_data import get_split


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


def get_model(
    saved_path="", pretrained_with_contrastive=False, pretrained_path="",
):
    if pretrained_with_contrastive:
        model = DeepMetricLearning()
    else:
        model = Classifier()
        model._build()
        model.load_weights(pretrained_path, by_name=True)

    os.makedirs(saved_path, exist_ok=True)
    return model


def get_callbacks(fold_id=0, saved_path=""):
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
        patience=30, monitor="val_lwlrap", mode="max"
    )
    return [model_checkpoint, early_stopping]


@cli.command("train-model", short_help="Train a Keras model.")
@click.option("--fold_idx", default=0, show_default=True)
@click.option("--saved_path", default="", show_default=True)
@click.option("--pretrained_path", default="", show_default=True)
@click.option("--pretrained_with_contrastive", default=0, show_default=True)
def main(fold_idx, saved_path, pretrained_path, pretrained_with_contrastive):
    train_data = pd.read_csv("./data/new_train_tp.csv")
    pretrained_with_contrastive = bool(pretrained_with_contrastive)

    for fold in range(5):
        if fold == fold_idx:
            os.makedirs(os.path.join(saved_path, f"fold{fold_idx}"), exist_ok=True)
            model = get_model(
                saved_path=saved_path,
                pretrained_with_contrastive=pretrained_with_contrastive,
                pretrained_path=pretrained_path,
            )

            # get train_idx and valid_idx
            train_index, val_index = get_split(fold=fold_idx)

            # convert to dictionary
            fold_train_dict = convert_csv_to_dict_for_dataloader(
                train_data.iloc[train_index]
            )
            fold_valid_dict = convert_csv_to_dict_for_dataloader(
                train_data.iloc[val_index]
            )

            # create dataloader
            balanced_train_data_loader = BalancedMelSampler(
                fold_train_dict,
                batch_size=64,
                max_length=NUM_FRAMES,
                is_train=True,
                n_classes=24,
                use_cutmix=True,
                cache=True,
                n_classes_in_batch=8,
                shuffle_aug=False,
            )

            valid_data_loader = MelSampler(
                fold_valid_dict,
                batch_size=balanced_train_data_loader.batch_size,
                n_classes=balanced_train_data_loader.n_classes,
                cache=True,
                max_length=NUM_FRAMES,
                is_train=False,
                use_cutmix=False,
                shuffle_aug=balanced_train_data_loader.shuffle_aug,
            )

            # build model, pass fake input.
            model._build()

            # compile model
            model.compile(
                optimizer=tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    tfa.optimizers.Lookahead(
                        tf.keras.optimizers.Adam(
                            learning_rate=tfa.optimizers.Triangular2CyclicalLearningRate(
                                initial_learning_rate=0.0001,
                                maximal_learning_rate=0.001,
                                step_size=50,
                            )
                        ),
                        10,
                        0.5,
                    ),
                    "dynamic",
                ),
                metrics=[TFLWLRAP(num_classes=24, name="lwlrap")],
                metric_loss_fn=NpairsLoss(temperature=0.1, name="n_pairs"),
                classification_loss_fn=tf.keras.losses.BinaryCrossentropy(
                    from_logits=True
                ),
            )

            # summary model
            model.summary()

            # training model.
            model.fit(
                balanced_train_data_loader,
                steps_per_epoch=int(
                    (len(fold_train_dict)) / balanced_train_data_loader.batch_size
                ),
                epoch=100,
                validation_data=valid_data_loader,
                callbacks=get_callbacks(fold_idx, saved_path=saved_path)
                if pretrained_with_contrastive is False
                else [
                    tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(saved_path, f"pretrained_best_fold{fold_idx}.h5"),
                        monitor="val_loss",
                        save_weights_only=True,
                        save_best_only=True,
                        mode="min",
                        save_freq="epoch",
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        patience=30, monitor="val_loss", mode="min",
                    ),
                ],
            )

            # clean model
            tf.keras.backend.clear_session()
            del model
            model = None
