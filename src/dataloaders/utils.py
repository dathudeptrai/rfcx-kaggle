import os
import numpy as np

from params import TRAIN_MELS_PATH, TEST_MELS_PATH


def csv_to_dict(csv_data):
    dict_data = {}

    for i in range(len(csv_data)):
        d = csv_data.iloc[i].tolist()
        if d[0] not in dict_data:
            dict_data[str(csv_data.iloc[i]["recording_id"])] = []
            dict_data[str(csv_data.iloc[i]["recording_id"])].append(d[1:])
            dict_data[str(csv_data.iloc[i]["recording_id"])].append(
                np.load(
                    os.path.join(
                        TRAIN_MELS_PATH,
                        str(csv_data.iloc[i]["raw_recording_id"]) + ".npy",
                    )
                )
            )
        else:
            dict_data[str(csv_data.iloc[i]["recording_id"])].append(d[1:])
            dict_data[str(csv_data.iloc[i]["recording_id"])].append(
                np.load(
                    os.path.join(
                        TRAIN_MELS_PATH,
                        str(csv_data.iloc[i]["raw_recording_id"]) + ".npy",
                    )
                )
            )

    return dict_data


def csv_to_dict_pl(csv_data):
    dict_data = {}

    for i in range(len(csv_data)):
        if csv_data["is_test"][i]:
            spec = np.load(TEST_MELS_PATH + csv_data["recording_id"][i] + ".npy")
        else:
            spec = np.load(TRAIN_MELS_PATH + csv_data["recording_id"][i] + ".npy")
        dict_data[csv_data["recording_id"][i]] = [spec, csv_data["pl"][i]]

    return dict_data
