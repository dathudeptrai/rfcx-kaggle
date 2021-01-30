import tensorflow as tf
import tensorflow_probability as tfp


class MixStyle(tf.keras.layers.Layer):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.beta = tfp.distributions.Beta(alpha, alpha)
        self.eps = eps

    def call(self, x, training=False):
        if training is False:
            return x

        if tf.random.uniform(shape=[], maxval=1.0) > self.p:
            return x

        B = tf.shape(x)[0]

        mu = tf.math.reduce_mean(x, axis=[1, 2], keepdims=True)
        var = tf.math.reduce_variance(x, axis=[1, 2], keepdims=True)
        sig = tf.math.sqrt((var + self.eps))
        mu = tf.stop_gradient(mu)
        sig = tf.stop_gradient(sig)
        x_norm = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = tf.cast(lmda, tf.float32)
        perm = tf.random.shuffle(tf.range(start=0, limit=B, dtype=tf.int32))
        mu2 = tf.gather(mu, perm)
        sig2 = tf.gather(sig, perm)

        mu_mix = mu * lmda + mu2 * (1.0 - lmda)
        sig_mix = sig * lmda + sig2 * (1.0 - lmda)

        return x_norm * sig_mix + mu_mix
