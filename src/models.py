import os
import tensorflow as tf
from backbones import ModelFactory


os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(42)

NUM_FRAMES = 512
NUM_FEATURES = 128
MODEL_FACTORY = ModelFactory()


class CBAMAttention(tf.keras.layers.Layer):
    def __init__(self, filters=1024, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()

        self.fc1 = tf.keras.layers.Dense(
            filters // ratio, activation="relu", use_bias=False
        )
        self.fc2 = tf.keras.layers.Dense(filters, activation="relu", use_bias=False)

        self.conv2d = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(7, 7), strides=1, padding="same", use_bias=False
        )

        self.filters = filters

    def call(self, inputs, training=False):
        avg_out = self.fc2(self.fc1(self.avg_pool(inputs)))
        max_out = self.fc2(self.fc1(self.max_pool(inputs)))
        channel = avg_out + max_out
        channel = tf.nn.sigmoid(channel)
        channel = tf.keras.layers.Reshape((1, 1, self.filters))(channel)
        channel_out = tf.multiply(inputs, channel)

        # spatial attention
        avg_pool = tf.reduce_mean(channel_out, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(channel_out, axis=-1, keepdims=True)
        spatial = tf.concat([avg_pool, max_pool], -1)
        spatial = self.conv2d(spatial)
        spatial_out = tf.nn.sigmoid(spatial)
        cbam_out = tf.multiply(channel_out, spatial_out)
        return cbam_out


class DeepMetricLearning(tf.keras.Model):
    def __init__(self, backbone_name="densenet121", **kwargs):
        super().__init__(**kwargs)
        self.backbone = MODEL_FACTORY.get_model_by_name(name=backbone_name)
        self.backbone._name = "backbone_global"

        self.interpolate = tf.keras.layers.UpSampling2D(size=(32, 1), interpolation='bilinear')
        self.cbam = CBAMAttention(self.backbone.output.shape[-1], ratio=8)

        self.fc = tf.keras.layers.Dense(units=128, name="fc")
        self.pooling = tf.keras.layers.GlobalAveragePooling2D(name="pooling")

    def _build(self):
        inputs = tf.keras.layers.Input(
            shape=[NUM_FRAMES, NUM_FEATURES, 3], dtype=tf.float32
        )
        self(inputs, training=True)

    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)  # bs x t x f x ft
        features = self.interpolate(features)  # bs x T x f x ft
        features = self.cbam(features)  # bs x T x f x ft

        features = self.pooling(features)
        features = self.fc(features)
        return features  # [B, 128]

    def compile(self, optimizer, metrics, metric_loss_fn, classification_loss_fn):
        super().compile(optimizer, metrics)
        self.metric_loss_fn = metric_loss_fn
        self.classification_loss_fn = classification_loss_fn

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
        x, y, _ = data

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
        self.pooling = None

        self.fc = tf.keras.layers.Dense(units=25, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)  # bs x t x f x ft
        features = self.interpolate(features)  # bs x T x f x ft
        features = self.cbam(features)  # bs x T x f x ft

        x = tf.reduce_mean(features, axis=2)  # bs x T x ft
        framewise_output = self.fc(x)[:, :, :24]  # bs x T x num_classes
        clipwise_output = tf.reduce_max(framewise_output, axis=1)
        return clipwise_output, framewise_output

    @tf.function
    def train_step(self, data):
        data = data[0]
        x, y, _ = data["x_tp"], data["y_tp"], data["y_seg_tp"]

        # forward step
        logits, seg_logits = self(x, training=True)

        # compute loss and calculate gradients
        cls_loss = self.classification_loss_fn(y, logits)
        # cls_loss = self.classification_loss_fn(y_seg, seg_logits)
        self._apply_gradients(cls_loss)

        self.metrics[0].update_state(y, logits)

        # return result
        results = {}
        results.update({"loss": cls_loss, "lwlrap": self.metrics[0].result()})
        return results

    @tf.function
    def test_step(self, data):
        x, y, y_seg = data
        logits, seg_logits = self(x, training=False)
        # compute loss and calculate gradients
        cls_loss = self.classification_loss_fn(y, logits)
        # cls_loss = self.classification_loss_fn(y_seg, seg_logits)

        self.metrics[0].update_state(y, logits)

        # return result
        results = {}
        results.update({"loss": cls_loss, "lwlrap": self.metrics[0].result()})
        return results
