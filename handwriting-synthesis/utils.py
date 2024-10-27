import random
import pickle
import config as c
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from network import Network
from dataset import buffer2array, OnlineHandwritingDataset
from loss import process_output_gaussian, process_output_sse

def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_writing(strokes, std_params, title=None, start=None, end=None, points=False, figsize=None, show=True):

    if figsize is not None:
        plt.figure(figsize=figsize)
    
    if start is None:
        start = 0

    if end is None:
        end = strokes.shape[0]
    
    data = strokes[start:end, :]

    for i in range(2):
        data[:, i] = np.cumsum(data[:, i] * std_params[i, 1] + std_params[i, 0], axis=0)

    data = np.split(data, 
                    np.where(data[:, -1] == 1)[0] + 1)

    for d in data:
        plt.plot(d[:, 0], -d[:, 1], color="black")
        if points:
            plt.scatter(d[:, 0], -d[:, 1], color="black")

    plt.axis("off")
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def load_standarization_params(standarization_file="handwriting-synthesis/model/standarization.csv"):
    with open(standarization_file, "rb") as f:
        params = buffer2array(f)
    return params

def encode_transcription(corpus, transcription):
    return [str(corpus.get(char, -1)) for char in transcription]

def decode_transcription(inverse_corpus, char_indexes):
    return "".join([inverse_corpus.get(char, "[UNK]") for char in char_indexes])

@tf.function
def generate_point_gaussian(output, smoothness=None):
    batch_size = output.shape[0]

    p, weights, means, std_dev, corr = process_output_gaussian(output, logits=True, b=smoothness)
    
    means = means[:, 0, ...]
    std_dev = std_dev[:, 0, ...]
    corr = corr[:, 0, ...]
    p = tf.concat([tf.zeros((batch_size, 1)), p], axis=-1)

    # sample one distribution for each batch member
    distr = tf.random.categorical(weights[:, 0, :], 1)
    distr = tf.reshape(distr, (batch_size, 1))

    # get parameters of sampled distribution
    means = tf.gather_nd(means, distr, batch_dims=1)
    std_dev = tf.gather_nd(std_dev, distr, batch_dims=1)
    corr = tf.gather(corr, tf.reshape(distr, (batch_size,)), batch_dims=1)
    cov = corr * tf.math.reduce_prod(std_dev, axis=-1)

    # build covariance matrix for each batch member and get its Cholesky decomposition matrix
    std_dev **= 2
    cov_matrix = tf.stack([tf.stack([std_dev[:, 0], cov], axis=0), tf.stack([cov, std_dev[:, 1]], axis=0)], axis=0)
    cov_matrix = tf.transpose(cov_matrix, (2, 0, 1))
    L = tf.linalg.cholesky(cov_matrix)
    
    # get next point offset coordinates
    point = tf.reshape(L @ tf.random.normal((batch_size, 2, 1)), (batch_size, 2)) + means
    point = tf.concat([point, tf.cast(tf.random.categorical(p, 1), dtype=np.float32)], axis=-1)
    return point[:, np.newaxis, :]

def generate_point_sse(output, smoothness=None):
    p, offsets = process_output_sse(output, logits=True)
    p = tf.concat([tf.zeros((output.shape[0], 1)), p], -1)
    point = tf.concat([offsets[:, 0, :], tf.cast(tf.random.categorical(p, 1), dtype=tf.float32)], 1)
    return point[:, np.newaxis, :]

def update_finish_idx(attention_idx, transcriptions_length, current_finish_idx, iteration):
    mask = tf.math.logical_and(attention_idx >= transcriptions_length + c.last_index_offset, current_finish_idx == -1.0)
    mask = tf.cast(mask, dtype=tf.float32)
    return current_finish_idx * (1 - mask) + iteration * mask

def check_finished(finish_idx):
    return tf.math.reduce_all(finish_idx != -1.0)

def clean_finish_idx(finish_idx):
    mask = tf.cast(finish_idx == -1, tf.float32)
    finish_idx = finish_idx * (1 - mask) + tf.tile(tf.cast([c.max_steps_inference], tf.float32), (finish_idx.shape[0],)) * mask
    return tf.cast(finish_idx, tf.int32)

def generate_handwriting(model_path, raw_transcriptions, initial_states=None):
    if c.use_gaussian_loss:
        generate_point = generate_point_gaussian
    else:
        generate_point = generate_point_sse
    
    batch_size = len(raw_transcriptions)

    _, _, _, transcriptions = OnlineHandwritingDataset(tf.zeros((batch_size, 2, 3)), raw_transcriptions).get_batch(0)
    strokes = tf.zeros((batch_size, 1, 3))
    states = initial_states
    transcriptions_length = tf.constant([len(t) for t in raw_transcriptions], dtype=tf.float32)
    finish_idx = tf.tile([-1.0], (transcriptions_length.shape[0],))

    model = Network()
    model(strokes, transcriptions)
    model.load_weights(model_path)

    for i in range(c.max_steps_inference):
        output, attention_idx, states = model(strokes[:, -1, :][:, np.newaxis, :], transcriptions, 
                                              training=False, initial_state=states)
        point = generate_point(output, smoothness=c.smoothness)
        strokes = tf.concat([strokes, point], axis=1)

        attention_idx = attention_idx[:, 0, 0]
        finish_idx = update_finish_idx(attention_idx, transcriptions_length, finish_idx, i)

        if check_finished(finish_idx):
            break

    strokes = strokes.numpy()[:, 1:, :]
    finish_idx = clean_finish_idx(finish_idx)

    return strokes, finish_idx



if __name__ == "__main__":

    batch = 32
    timesteps = 700
    distributions = 20


    # test process output for gaussian mixture loss
    output = tf.random.normal((batch, timesteps, distributions*6 + 1))
    p, weights, means, std_dev, corr = process_output_gaussian(output)
    assert p.shape == (batch, timesteps)
    assert weights.shape == (batch, timesteps, distributions)
    assert means.shape == (batch, timesteps, distributions, 2)
    assert std_dev.shape == (batch, timesteps, distributions, 2)
    assert corr.shape == (batch, timesteps, distributions)

    # test process output for sse loss
    output2 = tf.random.normal((batch, timesteps, 3))
    p, coords = process_output_sse(output2)
    assert p.shape == (batch, timesteps)
    assert coords.shape == (batch, timesteps, 2)

    # # # plot chosen writing
    std_params = load_standarization_params()
    with open("handwriting-synthesis/model/inverse_corpus.pickle", "rb") as f:
        inverse_corpus = pickle.load(f)

    files = ["f02-000-02", "f01-042z-07", "d05-360z-01", "b06-008-08", "l04-461z-02", "a09-595z-04"]
    data_subsets = ["train", "train", "train", "dev", "dev", "dev"]

    for fname, data_sub in zip(files, data_subsets):
        with open(f"data/handwriting-dataset/{data_sub}/strokes/{fname}.csv", "rb") as f:
            strokes = buffer2array(f)

        with open(f"data/handwriting-dataset/{data_sub}/transcriptions/{fname}.csv", "rb") as f:
            title_idxs = buffer2array(f)[0]

        title = decode_transcription(inverse_corpus, title_idxs)
        plot_writing(strokes, std_params, title=title)

