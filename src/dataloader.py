import os
import math
import random
import numpy as np
import tensorflow as tf
import nlpaug.flow as naf
import nlpaug.augmenter.spectrogram as nas


from tqdm import tqdm
from sklearn.utils import shuffle
from collections import defaultdict

from params import TRAIN_MELS_PATH


os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

HOP_SIZE_RATIO = 50
NUM_FEATURES = 128


def convert_csv_to_dict_for_dataloader(csv_data):
    dict_data = {}

    for i in tqdm(range(len(csv_data))):
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


class BalancedMelSampler(tf.keras.utils.Sequence):
    def __init__(
        self,
        dict_data,
        batch_size=64,
        max_length=384,
        is_train=True,
        n_classes=24,
        use_cutmix=False,
        cache=True,
        n_classes_in_batch=16,
        shuffle_aug=False,
    ):
        self.dict_data = dict_data
        self.batch_size = batch_size
        self.max_length = max_length

        self.augment = naf.Sequential(
            [
                nas.FrequencyMaskingAug(mask_factor=10),
                nas.TimeMaskingAug(mask_factor=10),
            ]
        )

        self.is_train = is_train
        self.n_classes = n_classes
        self.use_cutmix = use_cutmix
        self.cache = cache
        self.n_classes_in_batch = n_classes_in_batch
        self.shuffle_aug = shuffle_aug
        self.fp_samples = None
        self.tp_samples = defaultdict(list)
        self.non_apply_cutmix_recording_ids = []

        if self.cache:
            self._cache_samples(self.tp_samples, self.dict_data)

        if self.is_train:
            self.on_epoch_end()

    def _cache_samples(self, samples, dict_data):
        for k in tqdm(list(dict_data.keys())):
            if len(self.find_all_same_recording_id(k)) >= 2:
                self.non_apply_cutmix_recording_ids.append(k)
            for segment in [dict_data[k][0]]:
                label_segment = segment[0]
                samples[str(label_segment)].append(
                    {
                        "start_index": int(segment[2] * HOP_SIZE_RATIO),
                        "end_index": int(segment[4] * HOP_SIZE_RATIO),
                        "recording_id": k,
                    }
                )

    def find_all_same_recording_id(self, recording_id):
        real_recording_id = recording_id.split("_")[0]
        all_recording_ids = []
        for k in self.dict_data.keys():
            if str(real_recording_id) == str(k.split("_")[0]):
                all_recording_ids.append(k)
        return all_recording_ids

    def return_label_in_segment(self, recording_id, t_min, t_max):
        labels = []
        all_recording_ids = []
        for k in self.dict_data.keys():
            real_recording_id = k.split("_")[0]
            if str(real_recording_id) == str(recording_id.split("_")[0]):
                all_recording_ids.append(k)
        for recording_id in all_recording_ids:
            all_segments = [self.dict_data[recording_id][0]]
            for segment in all_segments:
                s = int(segment[2] * HOP_SIZE_RATIO)
                e = int(segment[4] * HOP_SIZE_RATIO)
                _len = e - s
                if (
                    (t_min in range(s + int(0.25 * _len), e - int(0.25 * _len)))
                    or (t_max in range(s + int(0.25 * _len), e - int(0.25 * _len)))
                    or (t_min <= s and t_max >= e)
                    or ((t_min >= s and t_max <= e) and ((t_max - t_min) >= 0.5 * _len))
                ):
                    labels.append(segment[0])
        return labels

    def on_epoch_end(self):
        if self.is_train:
            all_keys = list(self.tp_samples.keys())
            random.shuffle(all_keys)
            shuffle_tp_samples = [
                (key, shuffle(self.tp_samples[key])) for key in all_keys
            ]
            self.tp_samples = dict(shuffle_tp_samples)

    def __len__(self):
        return 1000

    def _getitem(
        self,
        samples,
        dict_data,
        batch_size,
        use_cutmix=False,
        class_idxs=None,
        shuffle_aug=False,
        chunk_len=None,
    ):
        all_class_idxs = list(samples.keys())
        if class_idxs is None:
            random_class_idxs = np.random.choice(
                all_class_idxs,
                size=self.n_classes_in_batch,
                replace=len(all_class_idxs) < self.n_classes_in_batch,
            )
        else:
            random_class_idxs = class_idxs
        n_samples_per_class = batch_size // self.n_classes_in_batch
        batch_r = []
        batch_x = []
        batch_s = []
        batch_e = []
        batch_y = []
        for class_idx in random_class_idxs:
            all_samples_per_class = samples[str(class_idx)]
            try:
                sampling_samples = np.random.choice(
                    all_samples_per_class,
                    size=n_samples_per_class,
                    replace=len(all_samples_per_class) < n_samples_per_class,
                )
            except Exception:
                sampling_samples = [all_samples_per_class[0]]

            for sample in sampling_samples:
                batch_r.append(sample["recording_id"])
                batch_x.append(dict_data[sample["recording_id"]][-1])
                batch_s.append(sample["start_index"])
                batch_e.append(sample["end_index"])
                batch_y.append(int(float(class_idx)))

        # random sampling here
        batch_x_aug = []
        batch_s_aug = []
        batch_e_aug = []
        batch_y_aug = []

        # random chunk length for sampling
        if self.max_length is not None:
            random_chunk_len = self.max_length if chunk_len is None else chunk_len
        else:
            random_chunk_len = (
                random.choice(np.array(range(2, 14)) * 32)
                if chunk_len is None
                else chunk_len
            )

        for i, (r, x, start, end) in enumerate(zip(batch_r, batch_x, batch_s, batch_e)):
            x_aug, s_aug, e_aug, extra_labels = self.random_sample(
                r,
                x,
                start,
                end,
                chunk_len=random_chunk_len,
                augment=self.is_train,
                shuffle_aug=shuffle_aug,
            )
            batch_x_aug.append(x_aug)
            batch_s_aug.append(s_aug)
            batch_e_aug.append(e_aug)
            batch_y_aug.append(extra_labels)

        batch_x_aug = np.reshape(np.array(batch_x_aug), (len(batch_x), -1, 128, 3))
        categorical_batch_y = np.zeros(
            shape=[len(batch_y_aug), self.n_classes], dtype=np.float32
        )
        for i, extra_label in enumerate(batch_y_aug):
            for label in extra_label:
                categorical_batch_y[i, label] = 1.0

        if use_cutmix and self.is_train and np.random.random() <= 0.5:
            batch_x_aug, categorical_batch_y = self._cutmix(
                batch_r,
                batch_x_aug,
                batch_s_aug,
                batch_e_aug,
                categorical_batch_y,
                random_chunk_len,
            )

            batch_x_aug = np.reshape(batch_x_aug, (len(batch_x_aug), -1, 128, 3))
            categorical_batch_y = np.reshape(
                categorical_batch_y, (len(categorical_batch_y), -1)
            )

        return batch_x_aug, categorical_batch_y, random_class_idxs

    def __getitem__(self, index):
        samples = {}
        (batch_x_aug_tp, categorical_batch_y_tp, random_class_idxs) = self._getitem(
            self.tp_samples,
            self.dict_data,
            self.batch_size,
            use_cutmix=self.use_cutmix,
            shuffle_aug=self.shuffle_aug,
        )
        samples["x_tp"] = batch_x_aug_tp
        samples["y_tp"] = categorical_batch_y_tp
        return samples

    def random_sample(
        self, r, x, start, end, chunk_len, augment=False, shuffle_aug=False
    ):
        if (end - start) == chunk_len:
            x_aug = x[start:end]
            s_aug = 0
            e_aug = chunk_len
            extra_labels = self.return_label_in_segment(r, start, end)
        elif (end - start) > chunk_len:
            mel_segment = x[start:end]
            s_random = random.choice(range(0, (end - start - chunk_len)))
            e_random = s_random + chunk_len
            x_aug = mel_segment[s_random:e_random]
            s_aug = 0
            e_aug = chunk_len
            extra_labels = self.return_label_in_segment(
                r, start + s_random, start + e_random
            )
        else:
            # chunk MUST include all mel_segment
            maximum_padding_left_right = chunk_len - (end - start)
            random_shift_left = random.choice(range(0, maximum_padding_left_right))
            if start < random_shift_left:
                random_shift_left = start

            if start - random_shift_left + chunk_len <= len(x):
                x_aug = x[
                    start - random_shift_left: start - random_shift_left + chunk_len
                ]
                s_aug = random_shift_left
                e_aug = s_aug + (end - start)
                extra_labels = self.return_label_in_segment(
                    r, start - random_shift_left, start - random_shift_left + chunk_len
                )
            else:
                shift_left_more = start - random_shift_left + chunk_len - len(x)
                random_shift_left += shift_left_more
                x_aug = x[
                    start - random_shift_left: start - random_shift_left + chunk_len
                ]
                s_aug = random_shift_left
                e_aug = s_aug + (end - start)
                extra_labels = self.return_label_in_segment(
                    r, start - random_shift_left, start - random_shift_left + chunk_len
                )

        # augment here
        x_aug_original = np.copy(x_aug)
        if self.is_train and np.random.random() <= 0.5 and augment:
            x_aug = self.augment.augment(x_aug)
        x_aug[s_aug:e_aug] = x_aug_original[s_aug:e_aug]

        if shuffle_aug and self.is_train:
            x_aug = self.remove_background(x_aug, s_aug, e_aug)

        return x_aug, s_aug, e_aug, extra_labels

    def remove_background(self, x_aug, s_aug, e_aug):
        x_aug_original = np.copy(x_aug)
        x_aug_shape = np.shape(x_aug)
        x_aug_flatten = np.reshape(x_aug, (-1))
        x_aug_flatten_shuffle = np.zeros_like(x_aug_flatten)
        x_aug = np.reshape(x_aug_flatten_shuffle, x_aug_shape)
        x_aug[s_aug:e_aug] = x_aug_original[s_aug:e_aug]
        return x_aug

    def _cutmix(self, r, x, s, e, y, random_chunk_len):
        x_aug = []
        y_aug = []
        for i in range(len(x)):
            original_sample = x[i]
            original_label = y[i]
            original_s = s[i]
            original_e = e[i]
            mixed_sample_idex = np.random.choice(
                [k for k in range(len(x)) if np.argmax(y[k]) != np.argmax(y[i])], size=1
            )[0]
            r_mixed_sample = r[mixed_sample_idex]
            if (
                (random_chunk_len - original_e) > original_s
                and r[i] not in self.non_apply_cutmix_recording_ids
                and (random_chunk_len - original_e) >= 32
            ):
                # right cutmix
                cutmix_sample = np.copy(original_sample)
                (
                    cutmix_sample[original_e:random_chunk_len],
                    _s,
                    _e,
                    _labels,
                ) = self.random_sample(
                    r_mixed_sample,
                    self.dict_data[r_mixed_sample][-1],
                    int(self.dict_data[r_mixed_sample][0][2] * 50),
                    int(self.dict_data[r_mixed_sample][0][4] * 50),
                    random_chunk_len - original_e,
                    augment=False,
                )
                mixed_label = np.zeros_like(original_label)
                for lab in _labels:
                    mixed_label[lab] = 1.0
                cutmix_label = np.clip(
                    np.array(original_label) + np.array(mixed_label), 0, 1
                )
                x_aug.append(cutmix_sample)
                y_aug.append(cutmix_label)
            elif (
                (random_chunk_len - original_e) < original_s
                and r[i] not in self.non_apply_cutmix_recording_ids
                and original_s >= 32
            ):
                # left cutmix
                cutmix_sample = np.copy(original_sample)
                cutmix_sample[0:original_s], _s, _e, _labels = self.random_sample(
                    r_mixed_sample,
                    self.dict_data[r_mixed_sample][-1],
                    int(self.dict_data[r_mixed_sample][0][2] * 50),
                    int(self.dict_data[r_mixed_sample][0][4] * 50),
                    original_s,
                    augment=False,
                )
                mixed_label = np.zeros_like(original_label)
                for lab in _labels:
                    mixed_label[lab] = 1.0
                cutmix_label = np.clip(
                    np.array(original_label) + np.array(mixed_label), 0, 1
                )
                x_aug.append(cutmix_sample)
                y_aug.append(cutmix_label)
            else:
                # class 23
                if original_s == 0 and (random_chunk_len - original_e) == 0:
                    cutmix_sample = np.copy(original_sample)
                    cutmix_length = np.random.randint(32, random_chunk_len // 2)
                    cutmix_start = np.random.randint(
                        0, random_chunk_len - cutmix_length
                    )
                    (
                        cutmix_sample[cutmix_start: cutmix_start + cutmix_length],
                        _s,
                        _e,
                        _labels,
                    ) = self.random_sample(
                        r_mixed_sample,
                        self.dict_data[r_mixed_sample][-1],
                        int(self.dict_data[r_mixed_sample][0][2] * 50),
                        int(self.dict_data[r_mixed_sample][0][4] * 50),
                        cutmix_length,
                        augment=False,
                    )
                    mixed_label = np.zeros_like(original_label)
                    for lab in _labels:
                        mixed_label[lab] = 1.0
                    cutmix_label = np.clip(
                        np.array(original_label) + np.array(mixed_label), 0, 1
                    )
                    x_aug.append(cutmix_sample)
                    y_aug.append(cutmix_label)

        return x_aug, y_aug


class MelSampler(tf.keras.utils.Sequence):
    def __init__(
        self,
        dict_data,
        batch_size=64,
        n_classes=24,
        cache=True,
        max_length=384,
        is_train=False,
        use_cutmix=False,
        shuffle_aug=False,
    ):
        self.dict_data = dict_data
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.cache = cache
        self.max_length = max_length
        self.is_train = is_train
        self.use_cutmix = use_cutmix
        self.shuffle_aug = shuffle_aug

        self.augment = naf.Sequential(
            [
                nas.FrequencyMaskingAug(mask_factor=10),
                nas.TimeMaskingAug(mask_factor=10),
            ]
        )

        self.x = []
        self.y = []

        self._cache_samples()

        if self.is_train:
            self.on_epoch_end()

    def _cache_samples(self):
        for k, v in tqdm(self.dict_data.items()):
            for segment in [self.dict_data[k][0]]:
                fmin = int(segment[3])
                fmax = int(segment[5])
                label_segment = segment[0]
                self.x.append(
                    {
                        "start_index": int(segment[2] * HOP_SIZE_RATIO),
                        "end_index": int(segment[4] * HOP_SIZE_RATIO),
                        "fmin": fmin,
                        "fmax": fmax,
                        "recording_id": k,
                    }
                )
                self.y.append(label_segment)

    def on_epoch_end(self):
        if self.is_train:
            self.x, self.y = shuffle(self.x, self.y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        # batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_r = [x["recording_id"] for x in batch_x]

        batch_x_aug = []
        batch_s_aug = []
        batch_e_aug = []
        batch_y_aug = []

        for sample in batch_x:
            x_aug, s_aug, e_aug, extra_labels = self.random_sample(
                sample["recording_id"],
                self.dict_data[sample["recording_id"]][-1],
                sample["start_index"],
                sample["end_index"],
                chunk_len=self.max_length,
                augment=self.is_train,
                shuffle_aug=self.shuffle_aug,
            )  # [NUM_FRAMES, 224, 3]
            batch_x_aug.append(x_aug)
            batch_s_aug.append(s_aug)
            batch_e_aug.append(e_aug)
            batch_y_aug.append(extra_labels)

        if len(batch_x_aug[0].shape) == 2:
            batch_x_aug = np.expand_dims(np.array(batch_x_aug), -1)
            batch_x_aug = np.concatenate(
                [batch_x_aug, batch_x_aug, batch_x_aug], axis=-1
            )
        else:
            batch_x_aug = np.reshape(
                np.array(batch_x_aug), (-1, self.max_length, NUM_FEATURES, 3)
            )

        categorical_batch_y = np.zeros(
            shape=[len(batch_y_aug), self.n_classes], dtype=np.float32
        )
        for i, extra_label in enumerate(batch_y_aug):
            for label in extra_label:
                categorical_batch_y[i, label] = 1.0

        if self.use_cutmix and self.is_train:
            batch_x_aug, categorical_batch_y = self._cutmix(
                batch_r,
                batch_x_aug,
                batch_s_aug,
                batch_e_aug,
                categorical_batch_y,
                self.max_length,
            )

            batch_x_aug = np.reshape(batch_x_aug, (len(batch_x_aug), -1, 128, 3))
            categorical_batch_y = np.reshape(
                categorical_batch_y, (len(categorical_batch_y), -1)
            )

        return batch_x_aug, categorical_batch_y

    def _cutmix(self, r, x, s, e, y, random_chunk_len):
        x_aug = []
        y_aug = []
        for i in range(len(x)):
            original_sample = x[i]
            original_label = y[i]
            original_s = s[i]
            original_e = e[i]
            mixed_sample_idex = np.random.choice(
                [k for k in range(len(x)) if np.argmax(y[k]) != np.argmax(y[i])], size=1
            )[0]
            mixed_sample = x[mixed_sample_idex]
            mixed_s = s[mixed_sample_idex]
            mixed_e = e[mixed_sample_idex]
            r_mixed_sample = r[mixed_sample_idex]
            if (random_chunk_len - original_e) > original_s:
                # right cutmix
                cutmix_sample = np.copy(original_sample)
                (
                    cutmix_sample[original_e:random_chunk_len],
                    _s,
                    _e,
                    _labels,
                ) = self.random_sample(
                    r_mixed_sample,
                    mixed_sample,
                    mixed_s,
                    mixed_e,
                    random_chunk_len - original_e,
                    augment=False,
                    shuffle_aug=self.shuffle_aug,
                )
                mixed_label = np.zeros_like(original_label)
                for lab in _labels:
                    mixed_label[lab] = 1.0
                cutmix_label = np.clip(
                    np.array(original_label) + np.array(mixed_label), 0, 1
                )
                x_aug.append(cutmix_sample)
                y_aug.append(cutmix_label)
            elif (random_chunk_len - original_e) < original_s:
                # left cutmix
                cutmix_sample = np.copy(original_sample)
                cutmix_sample[0:original_s], _s, _e, _labels = self.random_sample(
                    r_mixed_sample,
                    mixed_sample,
                    mixed_s,
                    mixed_e,
                    original_s,
                    augment=False,
                    shuffle_aug=self.shuffle_aug,
                )
                mixed_label = np.zeros_like(original_label)
                for lab in _labels:
                    mixed_label[lab] = 1.0
                cutmix_label = np.clip(
                    np.array(original_label) + np.array(mixed_label), 0, 1
                )
                x_aug.append(cutmix_sample)
                y_aug.append(cutmix_label)
            else:
                # Class 23
                if original_s == 0 and (random_chunk_len - original_e) == 0:
                    cutmix_sample = np.copy(original_sample)
                    cutmix_length = np.random.randint(32, random_chunk_len // 2)
                    cutmix_start = np.random.randint(
                        0, random_chunk_len - cutmix_length
                    )
                    (
                        cutmix_sample[cutmix_start: cutmix_start + cutmix_length],
                        _s,
                        _e,
                        _labels,
                    ) = self.random_sample(
                        r_mixed_sample,
                        self.dict_data[r_mixed_sample][-1],
                        int(self.dict_data[r_mixed_sample][0][2] * 50),
                        int(self.dict_data[r_mixed_sample][0][4] * 50),
                        cutmix_length,
                        augment=False,
                    )
                    mixed_label = np.zeros_like(original_label)
                    for lab in _labels:
                        mixed_label[lab] = 1.0
                    cutmix_label = np.clip(
                        np.array(original_label) + np.array(mixed_label), 0, 1
                    )
                    x_aug.append(cutmix_sample)
                    y_aug.append(cutmix_label)

        return x_aug, y_aug

    def return_label_in_segment(self, recording_id, t_min, t_max):
        labels = []
        all_recording_ids = []
        for k in self.dict_data.keys():
            real_recording_id = k.split("_")[0]
            if str(real_recording_id) == str(recording_id.split("_")[0]):
                all_recording_ids.append(k)
        for recording_id in all_recording_ids:
            all_segments = [self.dict_data[recording_id][0]]
            for segment in all_segments:
                s = int(segment[2] * HOP_SIZE_RATIO)
                e = int(segment[4] * HOP_SIZE_RATIO)
                _len = e - s
                if (
                    (t_min in range(s + int(0.25 * _len), e - int(0.25 * _len)))
                    or (t_max in range(s + int(0.25 * _len), e - int(0.25 * _len)))
                    or (t_min <= s and t_max >= e)
                    or ((t_min >= s and t_max <= e) and ((t_max - t_min) >= 0.5 * _len))
                ):
                    labels.append(segment[0])
        return labels

    def random_sample(
        self, r, x, start, end, chunk_len, augment=False, shuffle_aug=False
    ):
        if (end - start) == chunk_len:
            x_aug = x[start:end]
            s_aug = 0
            e_aug = chunk_len
            extra_labels = self.return_label_in_segment(r, start, end)
        elif (end - start) > chunk_len:
            mel_segment = x[start:end]
            s_random = random.choice(range(0, (end - start - chunk_len)))
            e_random = s_random + chunk_len
            x_aug = mel_segment[s_random:e_random]
            s_aug = 0
            e_aug = chunk_len
            extra_labels = self.return_label_in_segment(
                r, start + s_random, start + e_random
            )
        else:
            # chunk MUST include all mel_segment
            maximum_padding_left_right = chunk_len - (end - start)
            random_shift_left = random.choice(range(0, maximum_padding_left_right))
            # center crop for valid set
            # random_shift_left = int(maximum_padding_left_right // 2)
            if start < random_shift_left:
                random_shift_left = start

            if start - random_shift_left + chunk_len <= len(x):
                x_aug = x[
                    start - random_shift_left: start - random_shift_left + chunk_len
                ]
                s_aug = random_shift_left
                e_aug = s_aug + (end - start)
                extra_labels = self.return_label_in_segment(
                    r, start - random_shift_left, start - random_shift_left + chunk_len
                )
            else:
                shift_left_more = start - random_shift_left + chunk_len - len(x)
                random_shift_left += shift_left_more
                x_aug = x[
                    start - random_shift_left: start - random_shift_left + chunk_len
                ]
                s_aug = random_shift_left
                e_aug = s_aug + (end - start)
                extra_labels = self.return_label_in_segment(
                    r, start - random_shift_left, start - random_shift_left + chunk_len
                )

        if shuffle_aug and self.is_train:
            x_aug = self.remove_background(x_aug, s_aug, e_aug)

        return x_aug, s_aug, e_aug, extra_labels

    def remove_background(self, x_aug, s_aug, e_aug):
        x_aug_original = np.copy(x_aug)
        x_aug_shape = np.shape(x_aug)
        x_aug_flatten = np.reshape(x_aug, (-1))
        x_aug_flatten_shuffle = np.zeros_like(x_aug_flatten)
        x_aug = np.reshape(x_aug_flatten_shuffle, x_aug_shape)
        x_aug[s_aug:e_aug] = x_aug_original[s_aug:e_aug]
        return x_aug
