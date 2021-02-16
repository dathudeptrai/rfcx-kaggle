import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.sparse import csr_matrix


CLASS_WEIGHTS = np.array([
    0.029, 0.073, 0.020, 0.200, 0.013,
    0.034, 0.005, 0.078, 0.012, 0.009,
    0.005, 0.043, 0.078, 0.010, 0.043,
    0.043, 0.029, 0.012, 0.154, 0.003,
    0.018, 0.044, 0.007, 0.038
])
CLASS_WEIGHTS /= CLASS_WEIGHTS.sum()


def compute_indptr(y_true):
    y_true = csr_matrix(y_true)
    return y_true.indptr


def compute_y_true_indices(y_true):
    y_true = csr_matrix(y_true)
    return y_true.indices


def approx_rank(logits):
    list_size = tf.shape(input=logits)[1]
    x = tf.tile(tf.expand_dims(logits, 2), [1, 1, list_size])
    y = tf.tile(tf.expand_dims(logits, 1), [1, list_size, 1])
    pairs = tf.sigmoid((y - x) / 0.1)
    return tf.reduce_sum(input_tensor=pairs, axis=-1) + 0.5


def label_ranking_loss_tf(y_true, y_pred, is_fp=False):
    n_samples = tf.shape(y_true)[0]
    indptr = tf.py_function(compute_indptr, [y_true], Tout=tf.int32)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    y_true_indices = tf.py_function(compute_y_true_indices, [y_true], Tout=tf.int32)

    loss = 0.0
    for i in tf.range(n_samples):
        start = indptr[i]
        stop = indptr[1:][i]
        relevant = y_true_indices[start:stop]
        scores_i = y_pred[i]
        rank = approx_rank(tf.expand_dims(scores_i, 0))
        rank = tf.squeeze(rank, 0)
        rank = tf.gather(rank, relevant)
        L = tf.gather(scores_i, relevant)
        L = approx_rank(tf.expand_dims(L, 0))
        aux = tf.reduce_mean((L / rank))
        loss += aux

    loss = tf.math.divide(loss, tf.cast(n_samples, tf.float32))
    if is_fp:
        return loss
    else:
        return -1.0 * loss


class NpairsLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.05, name=None):
        super(NpairsLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )

        return tfa.losses.npairs_multilabel_loss(labels, logits)


class MovingAverageBCE(tf.keras.losses.Loss):
    def __init__(self, data_csv, start_apply_step=400, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.data_csv = data_csv
        r, labels = self._get_recording_id_and_label()
        self.moving_average_labels = tf.Variable(
            initial_value=labels,
            trainable=False,
            dtype=tf.float32,
            name="moving_average_labels",
        )
        self.labels = tf.Variable(
            initial_value=labels, trainable=False, dtype=tf.float32, name="labels",
        )
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.momentum = momentum
        self.r_to_idx = tf.keras.layers.experimental.preprocessing.StringLookup(
            num_oov_indices=0, vocabulary=r
        )
        self.start_apply_step = start_apply_step

    def _get_recording_id_and_label(self):
        r = []
        labels = []
        for i in range(len(self.data_csv)):
            row = self.data_csv.iloc[i]
            r.append(row["recording_id"])
            labels.append(row["species_id"])

        labels = tf.keras.utils.to_categorical(labels, num_classes=24)
        return r, labels

    def __call__(self, y_true, y_pred, recording_ids, iterations=0, is_cutmix=False):
        if (
            iterations <= tf.constant(self.start_apply_step, dtype=iterations.dtype)
            or is_cutmix
        ):
            bce = 0
            for i in range(len(CLASS_WEIGHTS)):
                bce += CLASS_WEIGHTS[i] * self.bce(y_true[:, i], y_pred[:, i])

            return tf.reduce_mean(bce)
        else:
            soft_labels = tf.stop_gradient(tf.nn.sigmoid(y_pred))
            index = self.r_to_idx(recording_ids) - 1  # 0 is oov
            for i in tf.range(len(index)):
                moving_average_pred = (
                    self.momentum * self.moving_average_labels[index[i]]
                    + (1.0 - self.momentum) * soft_labels[i]
                )
                moving_average_pred += self.labels[index[i]]
                moving_average_pred = tf.clip_by_value(moving_average_pred, 0.0, 1.0)
                self.moving_average_labels[index[i]].assign(moving_average_pred)

            y_true_update = tf.gather(self.moving_average_labels, index)

            bce = 0
            for i in range(len(CLASS_WEIGHTS)):
                bce += CLASS_WEIGHTS[i] * self.bce(y_true_update[:, i], y_pred[:, i])

            return tf.reduce_mean(bce)
