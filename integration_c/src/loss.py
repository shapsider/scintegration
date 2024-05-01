import tensorflow as tf
import random
import numpy as np
import tensorflow_probability as tfp


def replace_values_below_threshold(X, eps):
    return tf.where(X < eps, eps * tf.ones_like(X), X)


class BaseLoss:
    eps = 1e-9

    def __init__(self, model):
        self.n_output = len(list(model.clusters[0].trainable_variables))
        self.weight = 1

    @staticmethod
    def compute_distance(is_binary_input, output, target):
        """\
        Compute the distance between target and output with BCE if binary data or MSE for all others.
        """
        if is_binary_input:

            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(target, output, from_logits=True))
        else:

            return tf.reduce_mean(tf.square(target - output))


class SelfEntropyLoss(BaseLoss):
    def __init__(self, loss_weight=1.0):
        # super(SelfEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def __call__(self, cluster_outputs):
        eps = 1e-9
        cluster_outputs = tf.nn.softmax(cluster_outputs, axis=1)
        prob_mean = tf.reduce_mean(cluster_outputs, axis=0)
        prob_mean = tf.where(prob_mean < eps, eps, prob_mean)
        loss = -tf.reduce_sum(prob_mean * tf.math.log(prob_mean))

        loss *= self.loss_weight
        return loss

class DDCLoss(BaseLoss):
    def __init__(self, n_output, loss_weight=1.0):
        # super().__init__()
        self.weight = loss_weight
        self.n_output = n_output
        
        self.eye = tf.eye(self.n_output, dtype=tf.float32)

    @staticmethod
    def triu(X):

        return tf.reduce_sum(tf.linalg.band_part(X, 0, -1) - tf.linalg.band_part(X, 0, 0))

    @staticmethod
    def _atleast_epsilon(X, eps=1e-9):

        return tf.where(X < eps, eps * tf.ones_like(X), X)

    @staticmethod
    def cauchy_schwarz_divergence(A, K, n_clusters):

        nom = tf.transpose(A) @ K @ A
        dnom_squared = tf.expand_dims(tf.linalg.diag_part(nom), axis=-1) @ tf.expand_dims(tf.linalg.diag_part(nom), axis=0)

        nom = replace_values_below_threshold(nom, eps=BaseLoss.eps)
        dnom_squared = replace_values_below_threshold(dnom_squared, eps=BaseLoss.eps ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * tf.linalg.band_part(nom / tf.sqrt(dnom_squared), 0, -1)
        return d

    @staticmethod
    def d_cs(A, K, n_clusters):

        nom = tf.matmul(tf.matmul(tf.transpose(A), K), A)
        diag_nom = tf.linalg.diag_part(nom)
        dnom_squared = tf.expand_dims(diag_nom, -1) * tf.expand_dims(diag_nom, 0)


        nom = DDCLoss._atleast_epsilon(nom)
        dnom_squared = DDCLoss._atleast_epsilon(dnom_squared, eps=BaseLoss.eps ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * DDCLoss.triu(nom / tf.sqrt(dnom_squared))
        return d


    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):

        dist = tf.nn.relu(dist)
        sigma2 = rel_sigma * tfp.stats.percentile(dist, q=50)
        # Disable gradient for sigma
        # sigma2 = tf.stop_gradient(sigma2)
        sigma2 = tf.where(sigma2 < min_sigma, min_sigma * tf.ones_like(sigma2), sigma2)
        k = tf.exp(-dist / (2 * sigma2))
        return k


    @staticmethod
    def cdist(X, Y):
        xyT = X @ tf.transpose(Y)
        x2 =  tf.reduce_sum(X ** 2, axis=1, keepdims=True)
        y2 = tf.reduce_sum(Y ** 2, axis=1, keepdims=True)
        d = x2 - 2 * xyT + tf.transpose(y2)
        return d

    @staticmethod
    def vector_kernel(x, rel_sigma=0.15):

        return DDCLoss.kernel_from_distance_matrix(DDCLoss.cdist(x, x), rel_sigma)

    def __call__(self, hidden, cluster_outputs):
        loss = 0.

        cluster_outputs = tf.nn.softmax(cluster_outputs, axis=1)
        hidden_kernel = DDCLoss.vector_kernel(hidden)

        loss = DDCLoss.d_cs(cluster_outputs, hidden_kernel, self.n_output)

        m = tf.exp(-DDCLoss.cdist(cluster_outputs, self.eye))

        loss += DDCLoss.d_cs(m, hidden_kernel, self.n_output)
        loss *= self.weight

        return loss

if __name__ == "__main__":
    sce = SelfEntropyLoss(1.0)
    prob = tf.constant([[0.1, 0.9, 0.1], [0.1, 0.9, 0.1]], dtype=tf.float32)
    loss = sce(prob)
    print(loss.numpy())

    prob = tf.constant([[0.6, 0.6, 0.5], [0.4, 0.6, 0.7]], dtype=tf.float32)
    loss = sce(prob)
    print(loss.numpy())