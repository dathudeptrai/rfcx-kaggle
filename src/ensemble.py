import os
import glob
import click
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@click.group()
def cli():
    logging.basicConfig(
        format="%(asctime)12s - %(levelname)s - %(message)s", level=logging.INFO
    )


@cli.command("run-ensemble", short_help="Runing KFold multi-scale ensemble.")
@click.option("--checkpoints_path", default="", show_default=True)
def run_ensemble(checkpoints_path):
    # Step 1: Run multi-scale ensemble for each fold
    for fold in range(5):
        test_csv = pd.read_csv("../data/sample_submission.csv")
        lwlrap_at_scale = np.load(
            os.path.join(checkpoints_path, f"fold{fold}/lwlrap_at_scale.npy")
        )  # [24, n_scales]
        scale_csv_path = sorted(
            glob.glob(os.path.join(checkpoints_path, f"fold{fold}/*.csv"))
        )
        scale_predictions = [pd.read_csv(p) for p in scale_csv_path]
        preds = np.zeros(shape=[len(test_csv), 24], dtype=np.float32)

        for c in tqdm(range(24)):
            lwlrap_of_class = lwlrap_at_scale[c, :]
            best_scale = np.where(lwlrap_of_class == np.max(lwlrap_of_class))[0][
                -1:
            ]  # take a maximum scale.
            pred = np.zeros(shape=[len(test_csv)], dtype=np.float32)
            for scale in best_scale:
                pred_c_at_scale = np.array(
                    scale_predictions[scale]["s" + str(c)].tolist()
                )
                pred += pred_c_at_scale
            pred /= len(best_scale)
            preds[:, c] = pred

        for i in range(24):
            test_csv["s" + str(i)] = preds[:, i]

        test_csv.to_csv(
            os.path.join(checkpoints_path, f"fold{fold}/submission.csv"),
            index=False,
        )

    # Step 2: Kfold ensemble.
    all_csv_path = glob.glob(os.path.join(checkpoints_path, "**", "submission.csv"))
    fold_predictions = [pd.read_csv(p) for p in all_csv_path]
    test_csv = pd.read_csv("../data/sample_submission.csv")
    preds = np.zeros(shape=[len(test_csv), 24], dtype=np.float32)

    for i in tqdm(range(len(test_csv))):
        for k in range(len(fold_predictions)):
            preds[i] += tf.nn.sigmoid(fold_predictions[k].iloc[i].tolist()[1:]).numpy()
    preds /= len(fold_predictions)

    for i in range(24):
        test_csv["s" + str(i)] = preds[:, i]

    test_csv.to_csv("./submission.csv", index=False)


if __name__ == "__main__":
    cli()
