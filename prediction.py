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

from train import get_model

SCALES = [32, 64, 128, 192, 256, 320, 384, 448, 512]
NUM_FEATURES = 128


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


def generate_s_e_window_sliding(sample_len, win_size, step_size):
    start = 0
    end = win_size
    s_e = []
    s_e.append([start, end])
    while end < sample_len:
        start += step_size
        end = start + win_size
        s_e.append([start, end])

    s_e[-1][0] -= s_e[-1][1] - sample_len
    s_e[-1][1] = sample_len
    return s_e


@cli.command("run-prediction", short_help="test a trained keras model.")
@click.option("--checkpoints_path", default="", show_default=True)
@click.option("--fold", default=0, show_default=True)
def run_prediction(checkpoints_path, fold):
    test_csv = pd.read_csv("./data/sample_submission.csv")

    all_checkpoints = sorted(
        glob.glob(os.path.join(checkpoints_path, f"fold{fold}", "model-*.h5"))
    )

    model = get_model(
        saved_path=checkpoints_path,
        pretrained_with_contrastive=False,
        pretrained_path=all_checkpoints[-1],
    )

    preds = np.zeros((len(test_csv), 24), dtype=np.float32)

    @tf.function(experimental_relax_shapes=True)
    def predict(data):
        out = model(data, training=False)
        return out

    for s, scale in enumerate(SCALES):
        print("Start predicting for testset at scale: ", scale)
        for count, _ in enumerate(tqdm(test_csv["recording_id"].tolist())):
            k = test_csv.iloc[count]["recording_id"]
            mel = np.load(os.path.join("./data/test", k + ".npy"))
            mel_chunks = []
            list_s_e = generate_s_e_window_sliding(
                len(mel), win_size=scale, step_size=scale
            )  # consider to change step_size
            for (s, e) in list_s_e:
                mel_chunk = mel[s:e, ...]
                mel_chunks.append(mel_chunk)
            mel_chunks = np.array(mel_chunks).reshape((-1, scale, NUM_FEATURES, 3))
            chunk_preds = predict(mel_chunks).numpy()  # [n_chunks, 24]
            chunk_argmax = np.argmax(chunk_preds, axis=0)  # [24]
            score_each_row = []
            for i in range(24):
                score_each_row.append(chunk_preds[chunk_argmax[i], i])
            preds[count, :] = score_each_row

        for i in range(24):
            test_csv["s" + str(i)] = preds[:, i]

        test_csv.to_csv(
            os.path.join(checkpoints_path, f"fold{fold}", f"{s}.csv"), index=False,
        )


if __name__ == "__main__":
    cli()
