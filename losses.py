import numpy
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.sparse import csr_matrix


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
