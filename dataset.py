import math
import csv
import io
import re
import tarfile
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import config as c
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences


class OnlineHandwritingDataset(tf.keras.utils.Sequence):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.padding_kwargs = {"padding": "post", 
                               "truncating": "post", 
                               "dtype": "float32"}

    def __len__(self):
        # floor, cuz need only full batches for stateful lstm
        return math.floor(len(self.data) / c.batch_size)
    
    def __getitem__(self, idx):
        low = idx * c.batch_size
        high = min(low + c.batch_size, len(self.data))
        data = self.data[low:high]
        lengths = [len(d) - 1 for d in data]
        max_steps = min(np.max(lengths), c.max_steps)

        data = pad_sequences(data, maxlen=max_steps, **self.padding_kwargs)
        mask = tf.sequence_mask(lengths, maxlen=max_steps - 1)

        return data[:, 1:, :], data[:, 0:data.shape[1] - 1, :], mask
    
    def on_epoch_end(self):
        np.random.shuffle(self.data)

def load_standarization_params(standarization_file):
    with open(standarization_file) as f:
        params = f.readlines()[1]
        params = re.sub("\n$", "", params)
        params = params.split(",")
        params = [float(p) for p in params]
    return params

def plot_writing(strokes, standarization_file="model/standarization.txt", start=None, end=None, points=False):
    params = load_standarization_params(standarization_file)
    
    if start is None:
        start = 0

    if end is None:
        end = strokes.shape[0]
    
    data = strokes[start:end, :]

    data[:, 0:2] = np.cumsum(data[:, 0:2] * params[1] + params[0], axis=0)
    data = np.split(data, 
                    np.where(data[:, -1] == 1)[0] + 1)

    for d in data:
        plt.plot(d[:, 0], -d[:, 1], color="black")
        if points:
            plt.scatter(d[:, 0], -d[:, 1], color="black")
    plt.axis("off")
    plt.show()

def buffer2array(buffer):
    return np.asarray(list(csv.reader(io.TextIOWrapper(buffer))), dtype=float)

def load_data_from_tarfile(data_split_file, archive_name):
    with open(data_split_file) as f:
        file_names = f.readlines()

    file_names = [re.sub("\n$", "", fname) for fname in file_names]
    
    with tarfile.open(archive_name) as archive:
        data = [buffer2array(archive.extractfile(name)) for name in tqdm(file_names, ncols=c.tqdm_ncols)]

    return file_names, data

def load_data_from_folder(data_split_file, folder_name, n=None):
    print("Loading data...")

    with open(data_split_file) as f:
        file_names = f.readlines()

    file_names = [re.sub("\n$", "", fname) for fname in file_names]
    
    data = []
    for name in tqdm(file_names, ncols=c.tqdm_ncols):
        with open(folder_name + "/" + name, "rb") as f:
            data.append(buffer2array(f))
        if n is not None and len(data) == n:
            break

    print("Data loaded")
    
    return file_names, data

if __name__ == "__main__":

    # # dataset statistics
    lengths = []
    with tarfile.open("data/online_handwriting.tar.gz") as archive:
        for member in tqdm(archive.getmembers(), ncols=c.tqdm_ncols):
            if not ".csv" in member.name:
                continue

            f = archive.extractfile(member.name)
            lengths.append(buffer2array(f).shape[0])
    
    print("Num Observations:", len(lengths))
    print("Avg length:", np.mean(lengths))
    print("Med length:", np.median(lengths))
    plt.boxplot(lengths)
    plt.title("Lengths distribution")
    plt.show()  








