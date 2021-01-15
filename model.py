import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

tf.random.set_seed(42)


NUM_FRAMES = 384
NUM_FEATURES = 128


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


class AttBlock(tf.keras.layers.Layer):
    def __init__(self, filters, activation="softmax", temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.temperature = temperature
        self.att = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="same", use_bias=True
        )
        self.cla = tf.keras.layers.Conv1D(
            filters=filters + 1,  # ->>>> n_class + 1, class 25 is background
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
        )

    def call(self, x, training=False):
        # x ->> [B, T, F]
        norm_att = tf.nn.softmax(
            tf.clip_by_value(self.att(x), -10, 10), axis=1
        )  # [B, T, filters]
        cla = self.nonlinear_transform(self.cla(x))  # [B, T, filters + 1]
        cla = cla[:, :, :24]
        x = tf.reduce_sum(norm_att * cla, axis=1)  # [B, filters]
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "softmax":
            return tf.nn.softmax(x, -1)


class DeepMetricLearning(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone_global = tf.keras.applications.DenseNet121(include_top=False)
        self.cbam = CBAMAttention(self.backbone_global.output.shape[-1], ratio=8)
        self.fc = tf.keras.layers.Dense(units=128, activation="relu", name="fc")
        self.pooling = tf.keras.layers.GlobalAveragePooling2D(name="pooling")

    def _build(self):
        inputs = tf.keras.layers.Input(
            shape=[NUM_FRAMES, NUM_FEATURES, 3], dtype=tf.float32
        )
        self(inputs, training=True)

    def call(self, inputs, training=False):
        features = self.backbone_global(inputs, training=training)
        features = self.cbam(features)
        features = self.pooling(features)
        features = self.fc(features)
        return features  # [B, 128]

    def compile(self, optimizer, metrics, metric_loss_fn, classification_los_fn):
        super().compile(optimizer, metrics)
        self.metric_loss_fn = metric_loss_fn
        self.classification_los_fn = classification_los_fn

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
        features = self(x, training=True)

        # calculate metric learning loss
        metric_loss = tf.reduce_mean(self.metric_loss_fn(y, features))

        # return results
        results = {}
        results.update({"loss": metric_loss})
        return results


class Classifier(DeepMetricLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc, self.pooling = None
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.fc = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.att_block = AttBlock(filters=24, activation="softmax")

    def interpolate(self, x, ratio):
        upsampled = x[:, :, None, :]  # [B, T, 1, n_classes]
        upsampled = tf.tile(upsampled, [1, 1, ratio, 1])
        upsampled = tf.reshape(upsampled, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
        return upsampled

    def pad_framewise_output(self, framewise_output, frames_num):
        pad = tf.tile(
            framewise_output[:, -1:, :],
            [1, frames_num - tf.shape(framewise_output)[1], 1],
        )
        output = tf.concat([framewise_output, pad], axis=1)
        return output

    def call(self, inputs, training=False):
        features = self.backbone_global(inputs, training=training)
        features = self.cbam(features)  # [B, time, frequence, F]
        x = tf.reduce_mean(features, axis=2)  # [B, time, F]
        x = self.dropout(x, training=training)
        x = self.fc(x)  # [B, time, 512]
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        # Get framewise output
        framewise_output = self.interpolate(segmentwise_output, 32)
        framewise_output = self.pad_framewise_output(
            framewise_output, tf.shape(inputs)[1]
        )
        norm_att = self.interpolate(norm_att, 32)
        norm_att = self.pad_framewise_output(norm_att, tf.shape(inputs)[1])
        return clipwise_output, framewise_output, norm_att

    @tf.function
    def train_step(self, data):
        data = data[0]
        x_tp, y_tp = data["x_tp"], data["y_tp"]

        # forward step
        clipwise_output, *_ = self(x_tp, training=True)

        # compute loss and calculate gradients
        cls_loss = self.classification_loss_fn(y_tp, clipwise_output)
        self._apply_gradients(cls_loss)

        self.metrics[0].update_state(y_tp, clipwise_output)

        # return result
        results = {}
        results.update({"loss": cls_loss, "lwlrap": self.metrics[0].result()})
        return results

    @tf.function
    def test_step(self, data):
        x, y = data
        clipwise_output, *_ = self(x, training=True)
        # compute loss and calculate gradients
        cls_loss = self.classification_loss_fn(y, clipwise_output)

        self.metrics[0].update_state(y, clipwise_output)

        # return result
        results = {}
        results.update({"loss": cls_loss, "lwlrap": self.metrics[0].result()})
        return results
