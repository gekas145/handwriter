import pickle
import config as c
import utils as ut

model_path = "handwriting-synthesis/model/model.h5"
text_transcriptions = ["hello there",
                       "welcome to this repository",
                       "i can generate handwriting",
                       "just by looking at strings!"]

text_transcriptions = [t[0:c.max_transcription_length] for t in text_transcriptions]
with open("handwriting-synthesis/model/corpus.pickle", "rb") as f:
        corpus = pickle.load(f)

raw_transcriptions = [ut.encode_transcription(corpus, t) for t in text_transcriptions]

strokes, finish_idx = ut.generate_handwriting(model_path, raw_transcriptions)

std_params = ut.load_standarization_params()
for k in range(strokes.shape[0]):
    ut.plot_writing(strokes[k, :finish_idx[k], :], std_params, figsize=(8, 4))


