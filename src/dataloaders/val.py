import os
import math
import random
import numpy as np
import tensorflow as tf
import nlpaug.flow as naf
import nlpaug.augmenter.spectrogram as nas

from sklearn.utils import shuffle
from collections import defaultdict

from params import TRAIN_MELS_PATH, TEST_MELS_PATH


os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

HOP_SIZE_RATIO = 50
NUM_FEATURES = 128


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
        for k, v in self.dict_data.items():
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
        batch_y_seg_aug = []

        for sample in batch_x:
            x_aug, s_aug, e_aug, extra_labels, y = self.random_sample(
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
            batch_y_seg_aug.append(y)

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

        batch_y_seg_aug = np.array(batch_y_seg_aug)

        if self.use_cutmix and self.is_train:
            batch_x_aug, categorical_batch_y, batch_y_seg_aug = self._cutmix(
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

            batch_y_seg_aug = np.reshape(
                batch_y_seg_aug, (len(batch_y_seg_aug), -1, 24)
            )

        return batch_x_aug, categorical_batch_y, batch_y_seg_aug

    def _cutmix(self, r, x, s, e, y, y_seg, random_chunk_len):
        x_aug = []
        y_aug = []
        y_seg_aug = []
        for i in range(len(x)):
            original_sample = x[i]
            original_label = y[i]
            original_label_seg = y_seg[i]
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
                cutmix_y_seg = np.copy(original_label_seg)
                (
                    cutmix_sample[original_e:random_chunk_len],
                    _s,
                    _e,
                    _labels,
                    cutmix_y_seg[original_e:random_chunk_len],
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
                y_seg_aug.append(cutmix_y_seg)

            elif (random_chunk_len - original_e) < original_s:
                # left cutmix
                cutmix_sample = np.copy(original_sample)
                cutmix_y_seg = np.copy(original_label_seg)
                (
                    cutmix_sample[0:original_s],
                    _s,
                    _e,
                    _labels,
                    cutmix_y_seg[0:original_s]
                ) = self.random_sample(
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
                y_seg_aug.append(cutmix_y_seg)
            else:
                # Class 23
                if original_s == 0 and (random_chunk_len - original_e) == 0:
                    cutmix_sample = np.copy(original_sample)
                    cutmix_y_seg = np.copy(original_label_seg)
                    cutmix_length = np.random.randint(32, random_chunk_len // 2)
                    cutmix_start = np.random.randint(
                        0, random_chunk_len - cutmix_length
                    )
                    (
                        cutmix_sample[cutmix_start: cutmix_start + cutmix_length],
                        _s,
                        _e,
                        _labels,
                        cutmix_y_seg[cutmix_start: cutmix_start + cutmix_length]
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
                    y_seg_aug.append(cutmix_y_seg)

        return x_aug, y_aug, y_seg_aug

    def return_label_in_segment(self, recording_id, t_min, t_max):
        y = np.zeros((t_max - t_min, 24))

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

                frame_min = max(s - t_min, 0)
                frame_max = min(e - t_min, t_max - t_min)

                if frame_min < t_max - t_min and frame_max > 0:
                    y[frame_min:frame_max, segment[0]] = 1

        return labels, y

    def random_sample(
        self, r, x, start, end, chunk_len, augment=False, shuffle_aug=False
    ):
        if (end - start) == chunk_len:
            x_aug = x[start:end]
            s_aug = 0
            e_aug = chunk_len
            extra_labels, y = self.return_label_in_segment(r, start, end)
        elif (end - start) > chunk_len:
            mel_segment = x[start:end]
            s_random = random.choice(range(0, (end - start - chunk_len)))
            e_random = s_random + chunk_len
            x_aug = mel_segment[s_random:e_random]
            s_aug = 0
            e_aug = chunk_len
            extra_labels, y = self.return_label_in_segment(
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
                extra_labels, y = self.return_label_in_segment(
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
                extra_labels, y = self.return_label_in_segment(
                    r, start - random_shift_left, start - random_shift_left + chunk_len
                )

        if shuffle_aug and self.is_train:
            x_aug = self.remove_background(x_aug, s_aug, e_aug)

        return x_aug, s_aug, e_aug, extra_labels, y

    def remove_background(self, x_aug, s_aug, e_aug):
        x_aug_original = np.copy(x_aug)
        x_aug_shape = np.shape(x_aug)
        x_aug_flatten = np.reshape(x_aug, (-1))
        x_aug_flatten_shuffle = np.zeros_like(x_aug_flatten)
        x_aug = np.reshape(x_aug_flatten_shuffle, x_aug_shape)
        x_aug[s_aug:e_aug] = x_aug_original[s_aug:e_aug]
        return x_aug
