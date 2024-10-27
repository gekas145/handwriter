import pickle
import config as c
import utils as ut
from dataset import load_data
from utils import decode_transcription

model_path = "handwriting-synthesis/model/model.h5"
batch_id = 0
dataset_name = "dev"

with open("handwriting-synthesis/model/inverse_corpus.pickle", "rb") as f:
        inverse_corpus = pickle.load(f)

_, raw_transcriptions = load_data(dataset_name, start=batch_id * c.batch_size, end=(batch_id + 1) * c.batch_size)

strokes, finish_idx = ut.generate_handwriting(model_path, raw_transcriptions)

std_params = ut.load_standarization_params()
for k in range(strokes.shape[0]):
    ut.plot_writing(strokes[k, :finish_idx[k], :], std_params, figsize=(8, 4),
                    title=decode_transcription(inverse_corpus, raw_transcriptions[k]))


