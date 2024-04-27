import keras
import tensorflow as tf
import config as c

class Network(keras.Model):

    def __init__(self, stateful=True, **kwargs):
        super().__init__(**kwargs)
        self.lstm_layers = [keras.layers.LSTM(c.hidden_dim,
                                              stateful=stateful,
                                              return_sequences=True) for i in range(c.lstm_layers)]
        self.dense = keras.layers.Dense(c.output_dim)
        self.lstm_dropout = keras.layers.Dropout(c.lstm_dropout)
        self.dense_dropout = keras.layers.Dropout(c.dense_dropout)

    def call(self):
        pass

    def reset_states(self):
        for lstm in self.lstm_layers:
            lstm.reset_states()

    def _transform_output(self, output, training):
        if training:
            return output
        return output * self.std + self.mean

class Network1(Network):

    def call(self, inputs, mask=None, training=True):
        x = self.lstm_layers[0](inputs, mask=mask)
        for i in range(1, c.lstm_layers):
            x = self.lstm_dropout(x, training=training)
            x = self.lstm_layers[i](x, mask=mask)
        x = self.dense_dropout(x, training=training)
        return self.dense(x)
    
class Network2(Network):

    def call(self, inputs, mask=None, training=True):
        x = self.lstm_layers[0](inputs, mask=mask)
        hidden_states = [tf.identity(x)]
        for i in range(1, c.lstm_layers):
            x = clip_gradients(hidden_states[-1], c.clip_lstm)
            x = self.lstm_dropout(x, training=training)
            x = tf.concat([x, inputs], axis=-1)
            x = self.lstm_layers[i](x, mask=mask)
            hidden_states.append(tf.identity(x))

        x = tf.concat(hidden_states, axis=-1)
        x = self.dense_dropout(x, training=training)
        x = self.dense(x)
        return clip_gradients(x, c.clip_outputs)



@tf.custom_gradient
def clip_gradients(y, clip_value):
  def backward(dy):
    return tf.clip_by_value(dy, -clip_value, clip_value), None
  return y, backward


def process_output(output, logits=False):
    # important dimensions
    nbatch = output.shape[0]
    steps = output.shape[1]

    # extract distributions parameters
    p = output[..., 0]
    if not logits:
        p = tf.sigmoid(p)

    params = tf.reshape(output[..., 1:], (nbatch, steps, c.ndistr, 6))
    
    weights = params[..., 0]
    if not logits:
        weights = tf.nn.softmax(weights, axis=-1)
    
    means = params[..., 1:3]
    std_dev = tf.exp(params[..., 3:5])
    corr = tf.tanh(params[..., 5])

    return p, weights, means, std_dev, corr

def process_output2(output, logits=False):
    p = output[..., 2]
    if not logits:
        p = tf.sigmoid(p)

    return p, output[..., :2]


def process_output3(output, logits=False):
    nbatch = output.shape[0]
    steps = output.shape[1]

    p = output[..., 0]
    if not logits:
        p = tf.sigmoid(p)

    weights = output[..., 1:c.ndistr+1]
    if not logits:
        weights = tf.nn.softmax(weights, axis=-1)

    return p, weights, tf.reshape(output[..., c.ndistr+1:], (nbatch, steps, 2, c.ndistr))


