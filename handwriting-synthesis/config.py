import tensorflow as tf

# this file contains all main parameters regarding model and its training process

# model params
dense_dropout = 0.3
lstm_dropout = 0.3
n_distr = 20
n_mixtures = 10
use_gaussian_loss = True # set to False to use sse loss
hidden_size = 400
custom_rnn_dense_kernel_initializer_kwargs = {"minval": -0.001, "maxval": 0.001}
custom_rnn_dense_bias_initializer_kwargs = {"minval": -5.0, "maxval": -1.0}
input_size = 3 # do not edit this line
corpus_size = 58 # do not edit this line
output_size = 6 * n_distr + 1 if use_gaussian_loss else 3 # do not edit this line

# training data
max_steps = 700
test_frac = 0.06
max_transcription_length = 30

# training
epochs = 100
batch_size = 32
test_batch_size = 256
max_batches_per_epoch = None
lr = 0.001
clip_lstm_grad = tf.cast(10, tf.float32)
clip_outputs_grad = tf.cast(100, tf.float32)

# inference
max_steps_inference = 1000
last_index_offset = 0
smoothness = 1.5

# misc settings
tqdm_ncols = 80

