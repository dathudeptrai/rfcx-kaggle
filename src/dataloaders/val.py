import math
import numpy as np
import tensorflow as tf

from params import HOP_SIZE_RATIO, NUM_FEATURES


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

        self.x = []
        self.y = []

        self._cache_samples()

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
        pass

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]

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

        return batch_x_aug, categorical_batch_y, batch_y_seg_aug

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
            s_random = (end - start - chunk_len) // 2
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
            random_shift_left = maximum_padding_left_right // 2
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

        return x_aug, s_aug, e_aug, extra_labels, y
