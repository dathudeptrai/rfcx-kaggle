import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

from backbones import ModelFactory

tf.random.set_seed(42)


NUM_FRAMES = 512
NUM_FEATURES = 128
MODEL_FACTORY = ModelFactory()


class DeepMetricLearning(tf.keras.Model):
    def __init__(self, backbone_name="densenet121", **kwargs):
        super().__init__(**kwargs)
        self.backbone = MODEL_FACTORY.get_model_by_name(name=backbone_name)
        self.fc = tf.keras.layers.Dense(units=128, activation="relu", name="fc")
        self.pooling = tf.keras.layers.GlobalAveragePooling2D(name="pooling")

    def _build(self):
        inputs = tf.keras.layers.Input(
            shape=[NUM_FRAMES, NUM_FEATURES, 3], dtype=tf.float32
        )
        self(inputs, training=True)

    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)
        features = self.pooling(features)
        features = self.fc(features)
        return features  # [B, 128]

    def compile(
        self,
        optimizer,
        metrics,
        metric_loss_fn,
        classification_los_fn,
        moving_average_bce,
    ):
        super().compile(optimizer, metrics)
        self.metric_loss_fn = metric_loss_fn
        self.classification_los_fn = classification_los_fn
        self.moving_average_bce = moving_average_bce

    def _apply_gradients(self, total_loss):
        # compute gradient
        if isinstance(
            self.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer
        ):
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)
        else:
            scaled_loss = total_loss
        scaled_gradients = tf.gradients(scaled_loss, self.trainable_variables)
        if isinstance(
            self.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer
        ):
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = scaled_gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def train_step(self, data):
        data = data[0]
        x_tp, y_tp = data["x_tp"], data["y_tp"]

        # forward pass
        features = self(x_tp, training=True)

        # calculate metric learning loss
        metric_loss = tf.reduce_mean(self.metric_loss_fn(y_tp, features))

        # apply gradients
        self._apply_gradients(metric_loss)

        # return results
        results = {}
        results.update({"loss": metric_loss})
        return results

    @tf.function
    def test_step(self, data):
        x, y = data

        # forward pass
        features = self(x, training=False)

        # calculate metric learning loss
        metric_loss = tf.reduce_mean(self.metric_loss_fn(y, features))

        # return results
        results = {}
        results.update({"loss": metric_loss})
        return results


class Classifier(DeepMetricLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.logits = tf.keras.layers.Dense(units=24, activation=None)

    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)
        x = self.pooling(features)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        x = self.dropout(x, training=training)
        x = self.logits(x)
        return x

    @tf.function
    def train_step(self, data):
        data = data[0]
        x_tp, y_tp = data["x_tp"], data["y_tp"]

        # forward step
        logits = self(x_tp, training=True)

        # compute loss and calculate gradients
        cls_loss = self.moving_average_bce(
            y_tp, logits, data["r"], self.optimizer.iterations, data["is_cutmix"][0]
        )
        self._apply_gradients(cls_loss)

        self.metrics[0].update_state(y_tp, logits)

        # return result
        results = {}
        results.update({"loss": cls_loss, "lwlrap": self.metrics[0].result()})
        return results

    @tf.function
    def test_step(self, data):
        x, y = data
        logits = self(x, training=False)
        # compute loss and calculate gradients
        cls_loss = self.classification_loss_fn(y, logits)

        self.metrics[0].update_state(y, logits)

        # return result
        results = {}
        results.update({"loss": cls_loss, "lwlrap": self.metrics[0].result()})
        return results
