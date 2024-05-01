import tensorflow as tf
import tensorflow_probability as tfp
D = tfp.distributions

from src.num import EPS

class MSE(D.Distribution):

    def __init__(self, loc):
        super().__init__()
        self.loc = loc

    def log_prob(self, value):
        mse_loss = tf.keras.losses.MeanSquaredError()
        return -mse_loss(self.loc, value)

    @property
    def mean(self):
        return self.loc

class RMSE(MSE):

    def log_prob(self, value):
        return -tf.sqrt(tf.reduce_mean(tf.square(self.loc - value)))

class RMSE(MSE):

    def log_prob(self, value):
        return -tf.sqrt(tf.reduce_mean(tf.square(self.loc - value)))

class ZIN(D.Normal):

    def __init__(self, zi_logits, loc, scale):
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value):
        raw_log_prob = super().log_prob(value)
        zi_log_prob = tf.where(
            tf.abs(value) < EPS,
            tf.math.log(tf.exp(raw_log_prob) + tf.exp(self.zi_logits) + EPS) - tf.nn.softplus(self.zi_logits),
            raw_log_prob - tf.nn.softplus(self.zi_logits)
        )
        return zi_log_prob

class ZILN(D.LogNormal):

    def __init__(self, zi_logits, loc, scale):
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value):
        zi_log_prob = tf.where(
            tf.abs(value) < EPS,
            self.zi_logits - tf.nn.softplus(self.zi_logits),
            super().log_prob(value) - tf.nn.softplus(self.zi_logits)
        )
        return zi_log_prob

class ZINB(D.NegativeBinomial):

    def __init__(self, zi_logits, total_count, logits):
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value):
        raw_log_prob = super().log_prob(value)
        zi_log_prob = tf.where(
            tf.abs(value) < EPS,
            tf.math.log(tf.exp(raw_log_prob) + tf.exp(self.zi_logits) + EPS) - tf.nn.softplus(self.zi_logits),
            raw_log_prob - tf.nn.softplus(self.zi_logits)
        )
        return zi_log_prob




