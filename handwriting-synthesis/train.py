import tensorflow as tf
import config as c
from network import Network
from dataset import OnlineHandwritingDataset, load_data
from tqdm import tqdm
if c.use_gaussian_loss:
    from loss import gaussian_mixture_loss as loss_fn
else:
    from loss import sse_loss as loss_fn

@tf.function(
        input_signature=(tf.TensorSpec(shape=[c.test_batch_size, None, 3], dtype=tf.float32), 
                         tf.TensorSpec(shape=[c.test_batch_size, c.max_transcription_length, c.corpus_size], dtype=tf.float32), 
                         tf.TensorSpec(shape=[c.test_batch_size, None, 3], dtype=tf.float32), 
                         tf.TensorSpec(shape=[c.test_batch_size, None], dtype=tf.bool)))
def test_step(X, transcriptions, y_true, mask=None):
    y, _, _ = model(X, transcriptions, mask=mask, training=False)
    return loss_fn(y_true, y, mask=mask) * y_true.shape[0] / loss_scale

@tf.function(
        input_signature=(tf.TensorSpec(shape=[c.batch_size, None, 3], dtype=tf.float32), 
                         tf.TensorSpec(shape=[c.batch_size, c.max_transcription_length, c.corpus_size], dtype=tf.float32), 
                         tf.TensorSpec(shape=[c.batch_size, None, 3], dtype=tf.float32), 
                         tf.TensorSpec(shape=[c.batch_size, None], dtype=tf.bool)))
def train_step(X, transcriptions, y_true, mask=None):
    with tf.GradientTape() as tape:
        y, _, _ = model(X, transcriptions, mask=mask, training=True)
        loss = loss_fn(y_true, y, mask=mask)

    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

model = Network()

optimizer = tf.keras.optimizers.Adam(learning_rate=c.lr)
optimizer_checkpoint = tf.train.Checkpoint(optimizer=optimizer)

train_strokes, train_transcriptions = load_data("train")
dev_strokes, dev_transcriptions = load_data("dev")
train_dataset = OnlineHandwritingDataset(train_strokes, train_transcriptions)
train_dataset_length = train_dataset.get_length()
dev_dataset = OnlineHandwritingDataset(dev_strokes, dev_transcriptions)
dev_dataset_length = min(train_dataset.get_length(train=False), dev_dataset.get_length(train=False))

# clean up
del train_strokes, train_transcriptions, dev_strokes, dev_transcriptions

loss_scale = tf.cast(dev_dataset.get_length(train=False) * c.test_batch_size, tf.float32)
batches_per_epoch = c.max_batches_per_epoch if c.max_batches_per_epoch is not None else train_dataset_length

print("Start training")
batch = 0
for epoch in range(c.epochs+1):
    for _ in tqdm(range(batches_per_epoch), ncols=c.tqdm_ncols, leave=False):
        y_true, X, mask, transcriptions = train_dataset.get_batch(batch)

        train_step(X, transcriptions, y_true, mask=mask)
      
        batch += 1
        batch = batch % train_dataset_length
        if batch == 0:
            train_dataset.shuffle()
    
    loss_train, loss_dev = 0.0, 0.0
    for i in (pbar := tqdm(range(dev_dataset_length), ncols=c.tqdm_ncols, leave=False)):
        y_true, X, mask, transcriptions = train_dataset.get_batch(i, train=False)
        loss_train += test_step(X, transcriptions, y_true, mask=mask)

        y_true, X, mask, transcriptions = dev_dataset.get_batch(i, train=False)
        loss_dev += test_step(X, transcriptions, y_true, mask=mask)

        pbar.set_description(f"train/dev loss: {loss_train:.3f}/{loss_dev:.3f}")

    num = "0" if epoch < 10 else ""
    num += str(epoch)
    model.save_weights(f"checkpoints/{num} model train {loss_train:.3f} dev {loss_dev:.3f}.h5")
    optimizer_checkpoint.save(f"optimizers/{num} optimizer")

    print(f"[epoch {epoch}/{c.epochs}] loss train: {loss_train:.3f}, loss dev: {loss_dev:.3f}")


print("Training finished")



