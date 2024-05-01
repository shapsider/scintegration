import tensorflow as tf
import numpy as np

def scatter_add_2D(tensor, indices, updates):
    col_indices = tf.range(indices.shape[1])
    col_indices = tf.tile(tf.expand_dims(col_indices, 0), [indices.shape[0], 1])
    indices = tf.cast(indices, tf.int32)
    indexes = tf.stack([tf.reshape(indices, [-1]), tf.reshape(col_indices, [-1])], axis=1)
    updates = tf.reshape(updates, [-1])
    return tf.tensor_scatter_nd_add(tensor, indexes, updates)
            
class GraphConv(tf.keras.layers.Layer):
    def call(self, inputs, eidx, enorm, esgn):

        sidx, tidx = eidx  # source index and target index
        message =   tf.gather(inputs, sidx) * tf.expand_dims((esgn * enorm), -1)# n_edges * n_features

        indices = tf.expand_dims(tidx, 1)
        indices = tf.broadcast_to(indices, tf.shape(message))
        shape = tf.shape(inputs)
        res = scatter_add_2D(tf.zeros(shape), indices, message)
        return res
        
def get_default_numpy_dtype() -> type:
    tf_dtype_str = tf.keras.backend.floatx()
    if tf_dtype_str == 'float32':
        return np.float32
    elif tf_dtype_str == 'float64':
        return np.float64
    else:
        raise ValueError(f"Unsupported TensorFlow dtype: {tf_dtype_str}")
