import tensorflow as tf
import config as c
import numpy as np
from network import Network2
from dataset import OnlineHandwritingDataset, load_data_from_folder
from loss import mse_loss
from tqdm import tqdm

def get_pred_loss(data):
    return mse_loss(data[0], 
                    model(data[1], mask=data[2], training=False), 
                    data[2]) * data[0].shape[0]

def get_noise(data_shape):
    mask = tf.cast(np.random.binomial(1, c.noise_rate, data_shape[:2]), dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, (1, 1, 2))
    mask = tf.concat([mask, tf.zeros((*data_shape[:2], 1))], axis=-1)
    noise = np.random.normal(0, c.noise_var, data_shape)
    return noise * mask

model = Network2()
model.build(input_shape=(c.batch_size, None, c.input_dim))

optimizer = tf.keras.optimizers.Adam(learning_rate=c.lr)
optimizer_checkpoint = tf.train.Checkpoint(optimizer=optimizer)

_, train = load_data_from_folder("data_split/train.txt", "data/online_handwriting_train", 8000)
_, dev = load_data_from_folder("data_split/dev.txt", "data/online_handwriting_dev", 192)
train_dataset = OnlineHandwritingDataset(train)
dev_dataset = OnlineHandwritingDataset(dev)
del train, dev

loss_scale = len(dev_dataset) * c.batch_size
batches_per_epoch = min(c.max_batches_per_epoch, len(train_dataset))

batch = 0
for e in range(1, c.epochs+1):
    for _ in tqdm(range(batches_per_epoch), ncols=c.tqdm_ncols, leave=False):
        y_true, X, mask = train_dataset[batch]
        gradient = []

        for i in range(0, X.shape[1], c.train_steps):
            with tf.GradientTape() as tape:
                
                X_noise = tf.identity(X[:, i:i+c.train_steps, :])
                X_noise += get_noise(X_noise.shape)

                output = model(X_noise, mask=mask[:, i:i+c.train_steps])
                loss = mse_loss(y_true[:, i:i+c.train_steps, :], output, mask=mask[:, i:i+c.train_steps])

            if len(gradient) == 0:
                gradient = tape.gradient(loss, model.trainable_variables)
            else:
                g = tape.gradient(loss, model.trainable_variables)
                for j in range(len(gradient)):
                    gradient[j] += g[j]
        
        
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        model.reset_states()
        batch += 1
        if batch >= len(train_dataset):
            batch = 0
            train_dataset.on_epoch_end()
        
    
    loss_train, loss_dev = 0.0, 0.0
    for data_train, data_dev in zip(train_dataset, dev_dataset):
        loss_train += get_pred_loss(data_train)
        loss_dev += get_pred_loss(data_dev)

    loss_train = round(loss_train.numpy()/loss_scale, 2)
    loss_dev = round(loss_dev.numpy()/loss_scale, 2)

    # save to be able to continue later
    model.save(f"checkpoints/{e} model train {loss_train} dev {loss_dev}.keras")
    optimizer_checkpoint.save("optimizers/optimizer")

    print(f"[epoch {e}/{c.epochs}] loss train: {loss_train}, loss dev: {loss_dev}")




