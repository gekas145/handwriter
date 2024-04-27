import numpy as np
import config as c
import tensorflow as tf
from tensorflow import keras
from network import process_output2
from dataset import plot_writing, load_data_from_folder, OnlineHandwritingDataset

# unzip the model first
model = keras.models.load_model("model/model.keras")

# needs create_dataset.py to be run first
_, data = load_data_from_folder("data_split/train.txt", "data/online_handwriting_train", 15*c.batch_size)
dataset = OnlineHandwritingDataset(data)

num_init_steps = 50
y, X, mask = dataset[5]
strokes = X[:, 0:num_init_steps, :]

stroke_temp = 1.0

model(strokes, training=False)
for i in range(200):

    p, offsets = process_output2(model(strokes[:, -1, :][:, np.newaxis, :], training=False), logits=True)

    p = tf.concat([tf.zeros((c.batch_size, 1)), p/stroke_temp], -1)

    point = tf.concat([offsets[:, 0, :], tf.cast(tf.random.categorical(p, 1), dtype=tf.float32)], 1)

    strokes = tf.concat([strokes, point[:, np.newaxis, :]], 1)


for i in range(c.batch_size):
    plot_writing(strokes[i, num_init_steps:, :].numpy())


