import numpy as np
import tensorflow as tf


def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = retrieved_cumulative_hits[class_rankings[pos_class_indices]] / (
        1 + class_rankings[pos_class_indices].astype(np.float)
    )
    return pos_class_indices, precision_at_hits


class LwlrapAccumulator(object):
    """Accumulate batches of test samples into per-class and overall lwlrap."""

    def __init__(self):
        self.num_classes = 0
        self.total_num_samples = 0

    def accumulate_samples(self, batch_truth, batch_scores):
        """Cumulate a new batch of samples into the metric.

        Args:
          truth: np.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: np.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = np.zeros(self.num_classes)
            self._per_class_cumulative_count = np.zeros(self.num_classes, dtype=np.int)
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            (
                pos_class_indices,
                precision_at_hits,
            ) = _one_sample_positive_class_precisions(scores, truth)
            self._per_class_cumulative_precision[pos_class_indices] += precision_at_hits
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return self._per_class_cumulative_precision / np.maximum(
            1, self._per_class_cumulative_count
        )

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return self._per_class_cumulative_count / float(
            np.sum(self._per_class_cumulative_count)
        )

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return np.sum(self.per_class_lwlrap() * self.per_class_weight())


# from https://www.kaggle.com/carlthome/l-lrap-metric-for-tf-keras
@tf.function
def _tf_one_sample_positive_class_precisions(example):
    y_true, y_pred = example

    retrieved_classes = tf.argsort(y_pred, direction="DESCENDING")
    class_rankings = tf.argsort(retrieved_classes)
    retrieved_class_true = tf.gather(y_true, retrieved_classes)
    retrieved_cumulative_hits = tf.math.cumsum(
        tf.cast(retrieved_class_true, tf.float32)
    )

    idx = tf.where(y_true)[:, 0]
    i = tf.boolean_mask(class_rankings, y_true)
    r = tf.gather(retrieved_cumulative_hits, i)
    c = 1 + tf.cast(i, tf.float32)
    precisions = r / c

    dense = tf.scatter_nd(idx[:, None], precisions, [y_pred.shape[0]])
    return dense


class TFLWLRAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="lwlrap"):
        super().__init__(name=name)

        self._precisions = self.add_weight(
            name="per_class_cumulative_precision",
            shape=[num_classes],
            initializer="zeros",
        )

        self._counts = self.add_weight(
            name="per_class_cumulative_count", shape=[num_classes], initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        precisions = tf.map_fn(
            fn=_tf_one_sample_positive_class_precisions,
            elems=(y_true, y_pred),
            dtype=(tf.float32),
        )

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._precisions.assign_add(total_precisions)
        self._counts.assign_add(total_increments)

    def result(self):
        per_class_lwlrap = self._precisions / tf.maximum(self._counts, 1.0)
        per_class_weight = self._counts / tf.reduce_sum(self._counts)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_states(self):
        self._precisions.assign(self._precisions * 0)
        self._counts.assign(self._counts * 0)
