import random
import numpy as np
import tensorflow as tf
import nlpaug.flow as naf
import nlpaug.augmenter.spectrogram as nas

from sklearn.utils import shuffle
from collections import defaultdict

from params import HOP_SIZE_RATIO


class BalancedMelSampler(tf.keras.utils.Sequence):
    def __init__(
        self,
        dict_data,
        dict_data_fp,
        batch_size=64,
        max_length=384,
        n_classes=24,
        use_cutmix=False,
        n_classes_in_batch=16,
    ):
        self.dict_data = dict_data
        self.dict_data_fp = dict_data_fp
        self.batch_size = batch_size
        self.max_length = max_length

        self.specaugment = naf.Sequential(
            [
                nas.FrequencyMaskingAug(mask_factor=10),
                nas.TimeMaskingAug(mask_factor=10),
            ]
        )

        self.n_classes = n_classes
        self.use_cutmix = use_cutmix
        self.n_classes_in_batch = n_classes_in_batch
        self.fp_samples = None
        self.tp_samples = defaultdict(list)
        self.non_apply_cutmix_recording_ids = []

        self._cache_samples(self.tp_samples, self.dict_data)
        self.on_epoch_end()

    def _cache_samples(self, samples, dict_data):
        for k in list(dict_data.keys()):
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
        for k in self.dict_data_fp.keys():
            if str(real_recording_id) == str(k.split("_")[0]):
                all_recording_ids.append(k)
        return all_recording_ids

    def _return_label_in_segment(self, recording_id, t_min, t_max, fp=False):
        y = np.zeros((t_max - t_min, 24))

        labels = []
        all_recording_ids = []

        keys = self.dict_data_fp.keys() if fp else self.dict_data.keys()
        for k in keys:
            real_recording_id = k.split("_")[0]
            if str(real_recording_id) == str(recording_id.split("_")[0]):
                all_recording_ids.append(k)

        for recording_id in all_recording_ids:
            if fp:
                all_segments = [self.dict_data_fp[recording_id][0]]
            else:
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

    def return_label_in_segment(self, recording_id, t_min, t_max):
        labels, y = self._return_label_in_segment(recording_id, t_min, t_max, False)
        labels_fp, y_fp = self._return_label_in_segment(recording_id, t_min, t_max, True)

        return labels, y, labels_fp, y_fp

    def on_epoch_end(self):
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
        batch_y_seg_aug = []
        batch_y_aug_fp = []
        batch_y_seg_aug_fp = []

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
            x_aug, s_aug, e_aug, extra_labels, y, extra_labels_fp, y_fp = self.random_sample(
                r,
                x,
                start,
                end,
                chunk_len=random_chunk_len,
            )
            batch_x_aug.append(x_aug)
            batch_s_aug.append(s_aug)
            batch_e_aug.append(e_aug)
            batch_y_aug.append(extra_labels)
            batch_y_seg_aug.append(y)
            batch_y_aug_fp.append(extra_labels_fp)
            batch_y_seg_aug_fp.append(y_fp)

        batch_x_aug = np.reshape(np.array(batch_x_aug), (len(batch_x), -1, 128, 3))
        categorical_batch_y = np.zeros(
            shape=[len(batch_y_aug), self.n_classes], dtype=np.float32
        )
        for i, extra_label in enumerate(batch_y_aug):
            for label in extra_label:
                categorical_batch_y[i, label] = 1.0

        categorical_batch_y_fp = np.zeros(
            shape=[len(batch_y_aug_fp), self.n_classes], dtype=np.float32
        )
        for i, extra_label in enumerate(batch_y_aug_fp):
            for label in extra_label:
                categorical_batch_y[i, label] = 1.0

        batch_r = np.array(batch_r)
        batch_y_seg_aug = np.array(batch_y_seg_aug)
        batch_y_seg_aug_fp = np.array(batch_y_seg_aug_fp)

        mix = False
        use_cutmix = False
        if use_cutmix and np.random.random() <= 0.5:
            if np.random.random() <= 0.5:
                batch_x_aug, categorical_batch_y, batch_y_seg_aug = self._mixup(
                    batch_x_aug,
                    categorical_batch_y,
                    batch_y_seg_aug,
                )

                batch_x_aug = np.reshape(batch_x_aug, (len(batch_x_aug), -1, 128, 3))
                categorical_batch_y = np.reshape(
                    categorical_batch_y, (len(categorical_batch_y), -1)
                )

                batch_y_seg_aug = np.reshape(
                    batch_y_seg_aug, (len(batch_y_seg_aug), -1, 24)
                )

            batch_x_aug, categorical_batch_y, batch_y_seg_aug = self._cutmix(
                batch_r,
                batch_x_aug,
                batch_s_aug,
                batch_e_aug,
                categorical_batch_y,
                batch_y_seg_aug,
                random_chunk_len,
            )

            batch_x_aug = np.reshape(batch_x_aug, (len(batch_x_aug), -1, 128, 3))
            categorical_batch_y = np.reshape(
                categorical_batch_y, (len(categorical_batch_y), -1)
            )

            batch_y_seg_aug = np.reshape(
                batch_y_seg_aug, (len(batch_y_seg_aug), -1, 24)
            )
            mix = True

        return (
            batch_x_aug,
            categorical_batch_y,
            batch_y_seg_aug,
            categorical_batch_y_fp,
            batch_y_seg_aug_fp,
            batch_r,
            mix
        )

    def __getitem__(self, index):
        samples = {}
        (
            batch_x_aug_tp,
            categorical_batch_y_tp,
            batch_y_seg_aug,
            categorical_batch_y_fp,
            batch_y_seg_aug_fp,
            batch_r,
            is_cutmix
        ) = self._getitem(
            self.tp_samples,
            self.dict_data,
            self.batch_size,
            use_cutmix=self.use_cutmix,
        )

        samples["x_tp"] = batch_x_aug_tp
        samples["y_tp"] = categorical_batch_y_tp
        samples["y_seg_tp"] = batch_y_seg_aug
        samples["y_fp"] = categorical_batch_y_fp
        samples["y_seg_fp"] = batch_y_seg_aug_fp
        samples["r"] = batch_r
        samples["is_cutmix"] = np.array([is_cutmix])
        return samples

    def random_sample(
        self, r, x, start, end, chunk_len
    ):
        if (end - start) == chunk_len:
            x_aug = x[start:end]
            s_aug = 0
            e_aug = chunk_len
            extra_labels, y, extra_labels_fp, y_fp = self.return_label_in_segment(r, start, end)
        elif (end - start) > chunk_len:
            mel_segment = x[start:end]
            s_random = random.choice(range(0, (end - start - chunk_len)))
            e_random = s_random + chunk_len
            x_aug = mel_segment[s_random:e_random]
            s_aug = 0
            e_aug = chunk_len
            extra_labels, y, extra_labels_fp, y_fp = self.return_label_in_segment(
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
                extra_labels, y, extra_labels_fp, y_fp = self.return_label_in_segment(
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
                extra_labels, y, extra_labels_fp, y_fp = self.return_label_in_segment(
                    r, start - random_shift_left, start - random_shift_left + chunk_len
                )

        # augment here
        x_aug_original = np.copy(x_aug)
        if np.random.random() <= 0.5:
            x_aug = self.specaugment.augment(x_aug)
        x_aug[s_aug:e_aug] = x_aug_original[s_aug:e_aug]

        return x_aug, s_aug, e_aug, extra_labels, y, extra_labels_fp, y_fp

    def _cutmix(self, r, x, s, e, y, y_seg, y_fp, y_seg_fp, random_chunk_len):
        x_aug = []
        y_aug = []
        y_seg_aug = []
        y_aug_fp = []
        y_seg_aug_fp = []
        for i in range(len(x)):
            original_sample = x[i]
            original_label = y[i]
            original_label_seg = y_seg[i]
            original_label_fp = y_fp[i]
            original_label_seg_fp = y_seg_fp[i]
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
                cutmix_y_seg = np.copy(original_label_seg)
                cutmix_y_seg_fp = np.copy(original_label_seg_fp)
                (
                    cutmix_sample[original_e:random_chunk_len],
                    _s,
                    _e,
                    _labels,
                    cutmix_y_seg[original_e:random_chunk_len],
                    _labels_fp,
                    cutmix_y_seg_fp[original_e:random_chunk_len]
                ) = self.random_sample(
                    r_mixed_sample,
                    self.dict_data[r_mixed_sample][-1],
                    int(self.dict_data[r_mixed_sample][0][2] * 50),
                    int(self.dict_data[r_mixed_sample][0][4] * 50),
                    random_chunk_len - original_e,
                )
                mixed_label = np.zeros_like(original_label)
                for lab in _labels:
                    mixed_label[lab] = 1.0
                cutmix_label = np.clip(
                    np.array(original_label) + np.array(mixed_label), 0, 1
                )

                mixed_label_fp = np.zeros_like(original_label_fp)
                for lab in _labels_fp:
                    mixed_label_fp[lab] = 1.0
                cutmix_label_fp = np.clip(
                    np.array(original_label_fp) + np.array(mixed_label_fp), 0, 1
                )

                x_aug.append(cutmix_sample)
                y_aug.append(cutmix_label)
                y_seg_aug.append(cutmix_y_seg)
                y_aug_fp.append(cutmix_label_fp)
                y_seg_aug_fp.append(cutmix_y_seg_fp)

            elif (
                (random_chunk_len - original_e) < original_s
                and r[i] not in self.non_apply_cutmix_recording_ids
                and original_s >= 32
            ):
                # left cutmix
                cutmix_sample = np.copy(original_sample)
                cutmix_y_seg = np.copy(original_label_seg)

                (
                    cutmix_sample[0:original_s], _s, _e, _labels, cutmix_y_seg[0:original_s]
                ) = self.random_sample(
                    r_mixed_sample,
                    self.dict_data[r_mixed_sample][-1],
                    int(self.dict_data[r_mixed_sample][0][2] * 50),
                    int(self.dict_data[r_mixed_sample][0][4] * 50),
                    original_s,
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
                # class 23
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

    def _mixup(self, x, y, y_seg):
        x_aug = []
        y_aug = []
        y_seg_aug = []

        mix_index = np.random.permutation(len(x))

        for i in range(len(x)):
            alpha = np.random.beta(5, 5)

            mix_sample = alpha * x[i] + (1 - alpha) * x[mix_index[i]]
            mix_y = np.clip(y[i] + y[mix_index[i]], 0, 1)
            mix_y_seg = np.clip(y_seg[i] + y_seg[mix_index[i]], 0, 1)

            x_aug.append(mix_sample)
            y_aug.append(mix_y)
            y_seg_aug.append(mix_y_seg)

        return x_aug, y_aug, y_seg_aug
