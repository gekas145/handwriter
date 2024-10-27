import numpy as np
import config as c
import tensorflow as tf

eps = 10**(-5)

def gaussian_mixture_loss(y_true, output, mask=None):
    p, weights, means, std_dev, corr = process_output_gaussian(output, logits=False)
    # calculate normal densities
    loss = tf.tile(tf.expand_dims(y_true[..., 0:2], -2), (1, 1, c.n_distr, 1)) - means
    loss /= std_dev
    loss = tf.math.reduce_sum(loss**2, axis=-1) - 2 * corr * tf.math.reduce_prod(loss, axis=-1)
    corr = 1/(1 - corr**2)
    loss *= -corr/2
    loss = tf.math.sqrt(corr) * tf.math.exp(loss) / tf.math.reduce_prod(std_dev, axis=-1)
    # log gaussian mixture + log end of stroke mass function
    loss = tf.math.reduce_sum(weights * loss, axis=-1)
    loss = tf.math.log(loss + eps) + (1 - y_true[..., -1])*tf.math.log(1 - p + eps) + y_true[..., -1]*tf.math.log(p + eps)
    if mask is not None:
        loss = loss * tf.cast(mask, dtype=tf.float32)
    return -tf.math.reduce_sum(loss)/tf.cast(tf.shape(output)[0], dtype=tf.float32)


def sse_loss(y_true, output, mask=None):
    p, coords = process_output_sse(output, logits=False)

    loss = tf.math.reduce_sum((coords - y_true[..., :2])**2, axis=-1)
    loss -= (1 - y_true[..., -1]) * tf.math.log(1 - p + eps) + y_true[..., -1] * tf.math.log(p + eps)

    if mask is not None:
        loss = loss * tf.cast(mask, dtype=tf.float32)

    return tf.math.reduce_sum(loss)/tf.cast(tf.shape(output)[0], dtype=tf.float32)

def process_output_gaussian(output, logits=False, b=None):
    # important dimensions
    nbatch = output.shape[0]

    # extract distributions parameters
    p = output[..., 0]
    if not logits:
        p = tf.sigmoid(p)

    params = tf.reshape(output[..., 1:], (nbatch, -1, c.n_distr, 6))
    
    weights = params[..., 0]
    if b is not None:
        weights *= 1.0 + b
    if not logits:
        weights = tf.nn.softmax(weights, axis=-1)
    
    means = params[..., 1:3]
    std_dev = params[..., 3:5]
    if b is not None:
        std_dev -= b
    std_dev = tf.exp(std_dev)
    corr = tf.tanh(params[..., 5])

    return p, weights, means, std_dev, corr

def process_output_sse(output, logits=False, b=None):
    p = output[..., 2]
    if not logits:
        p = tf.sigmoid(p)

    return p, output[..., :2]


if __name__ == "__main__":

    batch = 32
    timesteps = 700
    distributions = 20

    def get_y_true():
        x1 = tf.convert_to_tensor(np.random.normal(size=(batch, timesteps, 2)))
        x2 = tf.convert_to_tensor(np.random.binomial(1, 0.2, (batch, timesteps, 1)).astype(float))
        y_true = tf.concat([x1, x2], -1)
        return tf.cast(y_true, np.float32)

    # test gaussian mixture loss
    output = tf.random.normal((batch, timesteps, distributions*6 + 1))
    y_true = get_y_true()
    print(f"Gaussian Mixture Loss: {gaussian_mixture_loss(y_true, output):.3f}")

    # test sse loss
    output2 = tf.random.normal((batch, timesteps, 3))
    y_true2 = get_y_true()
    print(f"SSE Loss: {sse_loss(y_true2, output2):.3f}")



