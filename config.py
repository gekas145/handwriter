
# model params
dense_dropout = 0.3
lstm_dropout = 0.2
ndistr = 20
input_dim = 3
hidden_dim = 512
lstm_layers = 2
output_dim = 3

# training data
max_steps = 700
test_frac = 0.06

# training
epochs = 60
batch_size = 32
max_batches_per_epoch = 125
lr = 0.005
clip_lstm = 1
clip_outputs = 10
train_steps = 100
noise_rate = 0.15
noise_var = 1.0
stroke_end_penalty = 1.5

# misc settings
tqdm_ncols = 50


