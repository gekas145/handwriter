import re
import io
import csv
import math
import pickle
import numpy as np
import config as c
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences


class OnlineHandwritingDataset(tf.keras.utils.Sequence):

    def __init__(self, strokes, transcriptions):
        super().__init__()
        self.strokes = strokes
        self.transcriptions = transcriptions
        self.__indexes = np.array(range(len(strokes)))
        self.padding_kwargs = {"padding": "post", 
                               "truncating": "post", 
                               "dtype": "float32"}

    def get_length(self, train=True):
        # return only full batches for simplicity
        batch_size = c.batch_size if train else c.test_batch_size
        return math.floor(len(self.strokes) / batch_size)
    
    def get_batch(self, idx, train=True):
        batch_size = c.batch_size if train else c.test_batch_size
        low = idx * batch_size
        high = low + batch_size

        strokes = [self.strokes[i] for i in self.__indexes[low:high]]
        transcriptions = [self.transcriptions[i] for i in self.__indexes[low:high]]

        # process 
        transcriptions = pad_sequences(transcriptions, value=-1.0, maxlen=c.max_transcription_length, **self.padding_kwargs)
        transcriptions = tf.one_hot(transcriptions, c.corpus_size, axis=-1)

        # process strokes
        lengths = [len(s) for s in strokes]
        max_steps = min(np.max(lengths), c.max_steps)
        strokes = pad_sequences(strokes, maxlen=max_steps, **self.padding_kwargs)
        strokes = tf.concat([tf.zeros((strokes.shape[0], 1, 3)), strokes], axis=1)
        mask = tf.sequence_mask(lengths, maxlen=max_steps)

        return strokes[:, 1:, :], strokes[:, 0:strokes.shape[1]-1, :], mask, transcriptions
    
    def shuffle(self):
        np.random.shuffle(self.__indexes)

def buffer2array(buffer, dtype=float):
    return np.asarray(list(csv.reader(io.TextIOWrapper(buffer))), dtype=dtype)

def load_data(data_subset, start=None, end=None, shuffle=False):
    print("Loading data...")

    with open(f"data/handwriting-dataset/data_split/{data_subset}.txt") as f:
        file_names = f.readlines()

    file_names = [re.sub("\n$", ".csv", fname) for fname in file_names]
    file_names[-1] += ".csv"

    if shuffle:
        file_names = [fname for fname in np.random.choice(file_names, len(file_names), replace=False)]

    start = 0 if start is None else start
    end = len(file_names) if end is None else end

    num_samples = end - start
    if num_samples <= 0:
        raise ValueError("end must be greater than start")
    if start >= len(file_names):
        raise ValueError(f"start can't exceed number of data samples: {len(file_names)} (0 based indexing)")
    if end <= 0:
        raise ValueError("end can't be lower than 1")
    
    if num_samples < len(file_names):
        print(f"Loading {num_samples} of {len(file_names)} samples")

    file_names = file_names[start:end]
    
    strokes = []
    transcriptions = []
    for name in tqdm(file_names, ncols=c.tqdm_ncols, total=min(num_samples, len(file_names))):
        with open(f"data/handwriting-dataset/{data_subset}/strokes/{name}", "rb") as f:
            strokes.append(buffer2array(f))

        with open(f"data/handwriting-dataset/{data_subset}/transcriptions/{name}", "rb") as f:
            transcriptions.append(buffer2array(f, dtype=int)[0])

    print("Data loaded")
    
    return strokes, transcriptions


if __name__ == "__main__":

    # # # test load_data
    test_data, _ = load_data("train", start=10, end=50)
    assert len(test_data) == 40
    _, test_data = load_data("dev", end=50)
    assert len(test_data) == 50
    test_data, _ = load_data("train", end=50, shuffle=True)
    assert len(test_data) == 50

    # # # train dataset statistics
    strokes, transcriptions = load_data("train")
    with open("handwriting-synthesis/model/inverse_corpus.pickle", "rb") as f:
        inverse_corpus = pickle.load(f)

    strokes_lengths = [s.shape[0] for s in strokes]
    transcriptions_lengths = [len(t) for t in transcriptions]
    lengths_relation = np.mean([sl/tl for sl, tl in zip(strokes_lengths, transcriptions_lengths)])

    transcriptions = np.concatenate([np.array(t) for t in transcriptions])
    chars, counts = np.unique(transcriptions, return_counts=True)
    chars_counts = sorted(zip(chars, counts), key=lambda x: x[1])
    
    print("Num Observations:", len(strokes_lengths))
    print("Avg stoke length:",np.round(np.mean(strokes_lengths)))
    print("Med stroke length:", np.median(strokes_lengths))
    print("Avg transcription length:", np.round(np.mean(transcriptions_lengths)))
    print("Med transcription length:", np.median(transcriptions_lengths))

    for x in chars_counts:
        print(f"{inverse_corpus.get(x[0], 'Unknown')} - {x[1]}")

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(8, 8)
    fig.suptitle(f"Handwriting dataset\napprox {lengths_relation:.0f} strokes steps per character")

    axs[0].boxplot(strokes_lengths)
    axs[0].set_title("Strokes lengths")
    axs[0].set_xticks([])

    axs[1].boxplot(transcriptions_lengths)
    axs[1].set_title("Transcriptions lengths")
    axs[1].set_xticks([])

    plt.show()