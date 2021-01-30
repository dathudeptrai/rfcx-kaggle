import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

import glob
import logging
import os

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import MelSampler, convert_csv_to_dict_for_dataloader
from metrics import LwlrapAccumulator
from split_data import get_split
from train import get_model

SCALES = [32, 64, 128, 192, 256, 320, 384, 448, 512]


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


@cli.command(
    "run-multi-scale-eval", short_help="eval multi-scale a trained keras model."
)
@click.option("--backbone_name", default="densenet121", show_default=True)
@click.option("--checkpoints_path", default="", show_default=True)
@click.option("--fold", default=0, show_default=True)
def run_multi_scale_eval(backbone_name, checkpoints_path, fold):
    """
    This function will compute each species's lrap at different scale.
    The idea is, for each species in test-set, we will use its predicted 
    value at the scale that maximize its lrap in eval set. in the case 
    that the maximum lrap value is achieved at many different scales, let take a
    maximum scale because the smaller scale, the more false positive samples.
    """
    train_data = pd.read_csv("./data/new_train_tp.csv")
    _, val_index = get_split(fold=fold)
    valid_dataset_csv = train_data.iloc[val_index]

    valid_data_loader = MelSampler(
        convert_csv_to_dict_for_dataloader(valid_dataset_csv),
        cache=True,
        batch_size=64,
        n_classes=24,
        is_train=False,
        use_cutmix=False,
        shuffle_aug=False,
        max_length=384,
    )
    all_checkpoints = sorted(
        glob.glob(os.path.join(checkpoints_path, f"fold{fold}", f"model-*.h5"))
    )

    model = get_model(
        backbone_name=backbone_name,
        saved_path=checkpoints_path,
        pretrained_with_contrastive=False,
        pretrained_path=all_checkpoints[-1],
    )

    lwlrap_at_scale = np.zeros(shape=[24, len(SCALES)], dtype=np.float32)

    for s, max_length in enumerate(SCALES):
        print("Runing evaluation at scale: ", max_length)
        valid_data_loader.max_length = max_length
        clip_preds = model.predict(valid_data_loader, verbose=1)  # [B, 24]
        clip_preds = tf.nn.sigmoid(clip_preds).numpy()

        # compute gts
        gts = []
        for i in range(len(valid_dataset_csv)):
            gts.append(valid_dataset_csv.iloc[i]["species_id"])
        gts = tf.keras.utils.to_categorical(gts, 24)

        # compute overal lwlrap at scale s
        lwlrap_calculator = LwlrapAccumulator()
        lwlrap_calculator.accumulate_samples(gts, clip_preds)
        overal_lwlrap = lwlrap_calculator.overall_lwlrap()
        print(f"Overal fold {fold} LWLRAP at scale {SCALES[s]} is: {overal_lwlrap:.3f}")

        # compute lrap of each species and save to file.
        for i in range(24):
            lwlrap_calculator = LwlrapAccumulator()
            lwlrap_calculator.accumulate_samples(
                gts[np.argmax(gts, -1) == i], clip_preds[np.argmax(gts, -1) == i]
            )
            overal_lwlrap = lwlrap_calculator.overall_lwlrap()
            print(f"Lrap for class {i} at scale {SCALES[s]}: {overal_lwlrap:.3f}")
            lwlrap_at_scale[i, s] = overal_lwlrap

    np.save(
        os.path.join(checkpoints_path, f"fold{fold}", "lwlrap_at_scale.npy"),
        lwlrap_at_scale,
    )


if __name__ == "__main__":
    cli()
