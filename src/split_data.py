import os
import numpy as np
import pandas as pd
import tensorflow as tf
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_split(fold=0):
    df = pd.read_csv("../data/new_train_tp.csv")
    table = (
        df.groupby("raw_recording_id")["species_id"]
        .apply(
            lambda x: tf.scatter_nd(
                tf.expand_dims(x, 1), np.ones_like(x), shape=[24]
            ).numpy()
        )
        .reset_index()
    )

    skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    idx_splits = list(
        skf.split(table.raw_recording_id, np.stack(table.species_id.to_numpy()))
    )
    splits = list(
        map(
            lambda xs: (
                table.raw_recording_id[xs[0]].to_numpy(),
                table.raw_recording_id[xs[1]].to_numpy(),
            ),
            idx_splits,
        )
    )
    train_idx = []
    valid_idx = []

    for i in range(len(df)):
        if df.iloc[i]["raw_recording_id"] in splits[fold][0]:
            train_idx.append(i)

    for i in range(len(df)):
        if df.iloc[i]["raw_recording_id"] in splits[fold][1]:
            valid_idx.append(i)

    return train_idx, valid_idx
