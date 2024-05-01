import collections
from abc import abstractmethod
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
D = tfp.distributions

from src.num import EPS
from src.nn import GraphConv
from src.prob import ZILN, ZIN, ZINB

class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, negative_slope=0.2, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.negative_slope = negative_slope

    def call(self, inputs):
        return tf.maximum(self.negative_slope * inputs, inputs)


class GraphEncoder(tf.keras.Model):
    def __init__(self, vnum, out_features, name="GraphEncoder"):
        super(GraphEncoder, self).__init__(name=name)
        self.vrepr = self.add_weight(shape=(vnum, out_features), trainable=True, name="vrepr", initializer="zeros")
        self.conv = GraphConv(name="conv")
        self.loc = tf.keras.layers.Dense(out_features, name="loc")
        self.std_lin = tf.keras.layers.Dense(out_features, name="std_lin")

    def call(self, eidx, enorm, esgn):
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = tf.nn.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)




class GraphDecoder(tf.keras.Model):
    def __init__(self, name="GraphDecoder"):
        super(GraphDecoder, self).__init__(name=name)
    
    def call(self, v, eidx, esgn):
        sidx, tidx = eidx
        logits = esgn * tf.reduce_sum(tf.gather(v, sidx) * tf.gather(v, tidx), axis=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(tf.keras.Model):

    def __init__(self, in_features, out_features, h_depth=2, h_dim=256, dropout=0.2, name="encoder"):
        super(DataEncoder, self).__init__(name=name)
        self.h_depth = h_depth
        self.seq_layers = []

        ptr_dim = in_features
        for layer in range(self.h_depth):
            self.seq_layers.append(tf.keras.layers.Dense(h_dim, name=f"dense_{layer}"))
            self.seq_layers.append(LeakyReLU())
            self.seq_layers.append(tf.keras.layers.BatchNormalization(name=f"bn_{layer}"))
            self.seq_layers.append(tf.keras.layers.Dropout(dropout))
            ptr_dim = h_dim

        self.loc = tf.keras.layers.Dense(out_features, name="loc")
        self.std_lin = tf.keras.layers.Dense(out_features, name="std_lin")

    def compute_l(self, x):
        # 抽象方法，需要在子类中实现
        raise NotImplementedError

    def normalize(self, x, l):
        # 抽象方法，需要在子类中实现
        raise NotImplementedError

    def call(self, x, xrep, lazy_normalizer=True, training=False):
        num_elements = tf.size(xrep).numpy().item()
        if num_elements:
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in self.seq_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                ptr = layer(ptr, training=training)
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                ptr = layer(ptr, training=training)
            else:
                ptr = layer(ptr, training=training)
        loc = self.loc(ptr)
        std = tf.nn.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):

    def compute_l(self, x):
        return None

    def normalize(self, x, l):
        return x


class NBDataEncoder(DataEncoder):

    TOTAL_COUNT = 1e4

    def compute_l(self, x):
        return tf.reduce_sum(x,axis=1, keepdims=True)

    def normalize(self, x, l):
        # 执行标准化操作
        normalized_x = x * (self.TOTAL_COUNT / l)
        return tf.math.log1p(normalized_x)


class DataDecoder(tf.keras.Model):

    def __init__(self, name=""): # pylint: disable=unused-argument
        super(DataDecoder, self).__init__(name=name)

    @abstractmethod
    def call(self, u, v, b, l):
        raise NotImplementedError  # pragma: no cover


class NormalDataDecoder(DataDecoder):

    def __init__(self, out_features: int, n_batches: int = 1, name=""):
        super().__init__(name=name)
        
        self.scale_lin = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True, name="scale_lin")
        self.bias =self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True, name="bias")
        self.std_lin = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True, name="std_lin")

    def call(self, u, v, b, l):
        scale = tf.nn.softplus(tf.gather(self.scale_lin, b))
        loc = tf.multiply(scale, u @ tf.transpose(v))  + tf.gather(self.bias, b)
        std = tf.nn.softplus(tf.gather(self.std_lin,b)) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    def __init__(self, out_features: int, n_batches: int = 1, name=""):
        super().__init__(name="name")
        self.zi_logits = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)

    def call(self, u, v, b, l):
        scale = tf.nn.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = tf.nn.softplus(self.std_lin[b]) + EPS
        return ZIN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    def __init__(self, out_features: int, n_batches: int = 1,name="name"):
        super().__init__(name="name")
        self.scale_lin = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)
        self.bias = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)
        self.zi_logits = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)
        self.std_lin = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)

    def call(self, u, v, b, l):
        scale = tf.nn.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = tf.nn.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):

    def __init__(self, out_features: int, n_batches: int = 1, name="name") -> None:
        super().__init__(name="name")
        self.scale_lin = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)
        self.bias = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)
        self.log_theta = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)

    def call(self, u, v, b, l):
        scale = tf.nn.softplus(tf.gather(self.scale_lin, b))
        logit_mu = scale * tf.matmul(u, v, transpose_b=True) + tf.gather(self.bias, b)
        mu = tf.nn.softmax(logit_mu, axis=1) * l
        log_theta = tf.gather(self.log_theta, b)

        return D.NegativeBinomial(
            total_count=tf.exp(log_theta),
            logits=tf.math.log(mu + EPS) - log_theta
        )


class ZINBDataDecoder(NBDataDecoder):

    def __init__(self, out_features: int, n_batches: int = 1,name="name") -> None:
        super().__init__(name="name")
        self.zi_logits = self.add_weight(shape=(n_batches, out_features), initializer="zeros",trainable=True)

    def call(self, u, v, b, l):
        scale = tf.nn.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = tf.nn.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class Discriminator(tf.keras.Model):

    def __init__(self, in_features, out_features, n_batches=0, h_depth=2, h_dim=256, dropout=0.2, name="Discriminator"):
        super(Discriminator, self).__init__(name=name)
        self.n_batches = n_batches
        self.seq_layers = []

        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            self.seq_layers.append(tf.keras.layers.Dense(h_dim))
            self.seq_layers.append(LeakyReLU())
            self.seq_layers.append(tf.keras.layers.Dropout(dropout))
            ptr_dim = h_dim

        self.pred = tf.keras.layers.Dense(out_features)

    def call(self, x, b):
        if self.n_batches:
            b_one_hot = tf.one_hot(b, depth=self.n_batches)
            x = tf.concat([x, b_one_hot], axis=1)

        for layer in self.seq_layers:
            x = layer(x)

        return self.pred(x)


class Prior(tf.keras.layers.Layer):
    def __init__(self, loc=0.0, std=1.0):
        super(Prior, self).__init__()
        self.loc = tf.Variable(initial_value=loc, dtype=tf.float32, trainable=False)
        self.std = tf.Variable(initial_value=std, dtype=tf.float32, trainable=False)

    def call(self):
        return D.Normal(self.loc, self.std)

def create_normal_distribution(loc=0.0, std=1.0):
    return tfp.distributions.Normal(loc=loc, scale=std)

class IndDataDecoder(DataDecoder):

    def __init__(  # pylint: disable=unused-argument
            self, in_features: int, out_features: int, n_batches: int = 1
    ) -> None:
        super().__init__()
        self.v = self.add_weight(shape=(out_features, in_features), initializer="zeros",trainable=True)

    def call(self, u, v, b, l):

        return super(IndDataDecoder, self).forward(u, self.v, b, l)


class IndNormalDataDocoder(IndDataDecoder, NormalDataDecoder):
    r"""
    Normal data decoder independent of feature latent
    """


class IndZINDataDecoder(IndDataDecoder, ZINDataDecoder):
    r"""
    Zero-inflated normal data decoder independent of feature latent
    """


class IndZILNDataDecoder(IndDataDecoder, ZILNDataDecoder):
    r"""
    Zero-inflated log-normal data decoder independent of feature latent
    """


class IndNBDataDecoder(IndDataDecoder, NBDataDecoder):
    r"""
    Negative binomial data decoder independent of feature latent
    """


class IndZINBDataDecoder(IndDataDecoder, ZINBDataDecoder):
    r"""
    Zero-inflated negative binomial data decoder independent of feature latent
    """