import time
import keras
import tensorflow as tf
import numpy as np
import config as c
from keras.preprocessing.sequence import pad_sequences
if c.use_gaussian_loss:
    from loss import gaussian_mixture_loss as loss_fn
else:
    from loss import sse_loss as loss_fn

class CustomRNNCell(keras.layers.Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.n_mixtures = c.n_mixtures
        self.dense = keras.layers.Dense(3*self.n_mixtures,
                                        kernel_initializer=tf.keras.initializers.RandomUniform(**c.custom_rnn_dense_kernel_initializer_kwargs),
                                        bias_initializer=tf.keras.initializers.RandomUniform(**c.custom_rnn_dense_bias_initializer_kwargs))
        self.lstm_cell = keras.layers.LSTMCell(c.hidden_size)
        self.state_size = (c.corpus_size, self.n_mixtures, c.hidden_size, c.hidden_size)

    def call(self, input_at_t, states_at_t, constants):
        '''
        states_at_t = [w_{t-1}, kappa_{t-1}, h_{t-1}, c_{t-1}]
        constants   = [transcriptions, enumerated_transcriptions]
        '''

        input = tf.concat((input_at_t, states_at_t[0]), axis=-1)

        y, lstm_states_at_t_plus_1 = self.lstm_cell(input, states_at_t[2:])

        b, U, N = constants[0].shape
        U += 1

        y = self.dense(y)
        y = tf.exp(y)
        y = tf.reshape(y, (b, 3, self.n_mixtures))

        kappa_at_t = y[:, 2, :]
        y = y[:, :2, :]

        kappa_at_t += states_at_t[1]
        kappa_at_t = tf.tile(tf.expand_dims(kappa_at_t, axis=-1), (1, 1, U))

        y = tf.tile(tf.expand_dims(y, axis=-1), (1, 1, 1, U))

        w = y[:, 0, ...] * tf.exp(-y[:, 1, ...] * (kappa_at_t - constants[1])**2)
        w = tf.math.reduce_sum(w, axis=-2)

        attention_index = tf.expand_dims(tf.cast(tf.argmax(w, axis=-1), tf.float32), axis=1)
        U -= 1
        w = w[:, :U]

        w = tf.tile(tf.expand_dims(w, axis=-1), (1, 1, N))
        w = tf.math.reduce_sum(w * constants[0], axis=-2)

        return tf.concat([lstm_states_at_t_plus_1[0], w, attention_index], axis=-1), (w, kappa_at_t[..., 0], *lstm_states_at_t_plus_1)
    

class Network(keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_rnn = keras.layers.RNN(CustomRNNCell(), return_sequences=True, zero_output_for_mask=True, return_state=True)
        self.lstm1 = keras.layers.LSTM(c.hidden_size, return_sequences=True, zero_output_for_mask=True, return_state=True)
        self.lstm2 = keras.layers.LSTM(c.hidden_size, return_sequences=True, zero_output_for_mask=True, return_state=True)
        self.dense = keras.layers.Dense(c.output_size)
        self.lstm_dropout = keras.layers.Dropout(c.lstm_dropout)
        self.dense_dropout = keras.layers.Dropout(c.dense_dropout)

    def call(self, strokes, transcriptions, mask=None, initial_state=None, training=True):

        if initial_state is None:
            initial_state = [None] * 3

        # prepare input for custom RNN
        indexes_dim = transcriptions.shape[1] + 1
        indexes = tf.range(1, indexes_dim + 1, dtype=np.float32)
        indexes = tf.tile(tf.expand_dims(indexes, 0), (strokes.shape[0], 1))
        indexes = tf.tile(indexes, (1, c.n_mixtures))
        indexes = tf.reshape(indexes, (strokes.shape[0], c.n_mixtures, indexes_dim))

        # pass through custom RNN
        output = self.custom_rnn(strokes, 
                                 constants=(transcriptions, indexes), 
                                 mask=mask, 
                                 initial_state=initial_state[0])
        
        internal_states = [output[1:]]
        output = output[0]
        attention_vectors = output[..., c.hidden_size:c.hidden_size + transcriptions.shape[-1]]
        attention_index = output[..., c.hidden_size + transcriptions.shape[-1]:]
        hidden_outputs = [output[..., :c.hidden_size]]
        
        # pass through 1st LSTM
        output = clip_gradient(hidden_outputs[-1], c.clip_lstm_grad)
        output = self.lstm_dropout(output, training=training)
        output = tf.concat([output, attention_vectors, strokes], axis=-1)
        output = self.lstm1(output, mask=mask, initial_state=initial_state[1])
        hidden_outputs.append(output[0])
        internal_states.append(output[1:])

        # pass through 2nd LSTM
        output = clip_gradient(hidden_outputs[-1], c.clip_lstm_grad)
        output = self.lstm_dropout(output, training=training)
        output = tf.concat([output, attention_vectors, strokes], axis=-1)
        output = self.lstm2(output, mask=mask, initial_state=initial_state[2])
        hidden_outputs.append(output[0])
        internal_states.append(output[1:])

        # pass through Dense and output
        output = tf.concat(hidden_outputs, axis=-1)
        output = self.dense_dropout(output, training=training)
        output = self.dense(output)

        return clip_gradient(output, c.clip_outputs_grad), attention_index, internal_states


@tf.custom_gradient
def clip_gradient(y, clip_value):
  def backward(dy):
    return tf.clip_by_value(dy, -clip_value, clip_value), None
  return y, backward


if __name__ == "__main__":
    
    def get_transcriptions(batch_size):
        def get_random():
            return np.random.randint(0, c.corpus_size, size=np.random.randint(10, 30, 1)[0])
    
        transcriptions = [get_random() for i in range(batch_size)]
        transcriptions = pad_sequences(transcriptions, 
                                    value=-1.0, 
                                    maxlen=c.max_transcription_length,
                                    padding="post",
                                    truncating="post",
                                    dtype="float32")
        return tf.one_hot(transcriptions, c.corpus_size, axis=-1)

    K = 10
    U = 30
    N = c.corpus_size
    b = c.batch_size
    h_dim = 400
    x_dim = 3
    steps = 700

    x1 = tf.convert_to_tensor(np.random.normal(size=(c.batch_size, steps, 2)))
    x2 = tf.convert_to_tensor(np.random.binomial(1, 0.2, (c.batch_size, steps, 1)).astype(float))
    y_true = tf.concat([x1, x2], -1)
    y_true = tf.cast(y_true, np.float32)

    x = tf.random.normal((b, steps, x_dim))
    transcriptions = get_transcriptions(c.batch_size)

    model = Network()

    # measure time required for single training step
    start = time.time()
    with tf.GradientTape() as tape:
        output, _, _ = model(x, transcriptions)
        loss = loss_fn(y_true, output)
    
    grad = tape.gradient(loss, model.trainable_variables)
    diff = time.time() - start

    print(f"Output shape: {output.shape}")
    print(f"Elapsed time: {diff}")
    print(f"Loss value: {loss:.2f}")
    model.summary() # has around 3.7M trainable params

    # test attention vectors and internal states
    _, attention_vectors, states = model(x, transcriptions, training=False)
    print(f"Attention vectors shape: {attention_vectors.shape}")
    print(attention_vectors[0, :, 0])

    print(f"Internal states: {len(states)}")
    print(f"Internal states of each rnn: {[len(s) for s in states]}")

    # test passing the initial_state argument
    model(x, transcriptions, training=False, initial_state=states)

    # test passing the batch of different size
    x1 = tf.random.normal((4, steps, x_dim))
    transcriptions1 = get_transcriptions(4)
    model(x1, transcriptions1)

    # test weights saving
    model.save_weights("checkpoints/test_weights.h5")

    model1 = Network()
    model1(x, transcriptions)
    model1.load_weights("checkpoints/test_weights.h5")

    y, y1 = model(x, transcriptions, training=False)[0].numpy(), model1(x, transcriptions, training=False)[0].numpy()
    np.testing.assert_allclose(y, y1)



