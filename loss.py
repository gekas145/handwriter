import math
import config as c
import tensorflow as tf
from network import process_output, process_output2, process_output3

eps = 10**(-6)

def gaussian_mixture_loss(y_true, output, mask=None):

    p, weights, means, std_dev, corr = process_output(output)

    # calculate normal densities
    X = tf.tile(tf.expand_dims(y_true[..., 0:2], -2), (1, 1, c.ndistr, 1)) - means
    std_dev_prod = 1/tf.math.reduce_prod(std_dev, axis=-1)
    X = tf.math.reduce_sum((X/std_dev)**2, axis=-1) - 2*corr*std_dev_prod*tf.math.reduce_prod(X, axis=-1)
    corr = 1/(1 - corr**2)
    X = -X/2*corr
    loss = 1/(2 * math.pi)*tf.math.sqrt(corr)*std_dev_prod*tf.math.exp(X)

    # log gaussian mixture + log end of stroke mass function
    loss = tf.math.reduce_sum(weights * loss, axis=-1)
    loss = tf.math.log(loss + eps) + (1 - y_true[..., -1])*tf.math.log(1 - p + eps) + y_true[..., -1]*tf.math.log(p + eps)

    if mask is not None:
        loss = loss * tf.cast(mask, dtype=tf.float32)

    return -tf.math.reduce_sum(loss)/tf.cast(tf.shape(output)[0], dtype=tf.float32)

def mse_loss(y_true, output, mask=None):

    p, offsets = process_output2(output)

    loss = tf.math.reduce_sum((offsets - y_true[..., :2])**2, axis=-1)
    loss -= (1 - y_true[..., -1])*tf.math.log(1 - p + eps) + c.stroke_end_penalty*y_true[..., -1]*tf.math.log(p + eps)

    if mask is not None:
        loss = loss * tf.cast(mask, dtype=tf.float32)
    
    return tf.math.reduce_sum(loss)/output.shape[0]


def mse_mixture_loss(y_true, output, mask=None):

    p, weights, offsets = process_output3(output)

    loss = (offsets - tf.tile(tf.expand_dims(y_true[..., :2], -1), (1, 1, 1, c.ndistr)))**2
    loss = tf.math.reduce_sum(loss, axis=-2)
    loss = tf.math.reduce_sum(loss * weights, axis=-1)
    loss -= (1 - y_true[..., -1])*tf.math.log(1 - p + eps) + c.stroke_end_penalty*y_true[..., -1]*tf.math.log(p + eps)

    if mask is not None:
        loss = loss * tf.cast(mask, dtype=tf.float32)
    
    return tf.math.reduce_sum(loss)/output.shape[0]

