import re
import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import config as c
from tqdm import tqdm
from utils import encode_transcription

def readlines(fname):
    with open(fname) as f:
        lines = f.readlines()

    return [re.sub("^\s*|\s*\n$", "", line) for line in lines]

def process_transcriptions_and_filenames(original_names, data_subset, corpus=None):

    load_names = []
    save_names = []
    transcriptions = []
    for name in tqdm(original_names, ncols=c.tqdm_ncols):

        lname = f"data/ascii/{name.split('-')[0]}/{re.sub('[a-z]+$', '', name)}/{name}"
        
        lines = readlines(lname + ".txt")
        lines = lines[lines.index("CSR:") + 2:]

        lname = re.sub("ascii", "lineStrokes", lname)
        
        for i in range(1, len(lines) + 1):
            
            # omit blank lines
            if re.match("^\s*\n*$", lines[i-1]):
                continue

            idx = str(i) if i > 9 else f"0{i}"

            load_names.append(f"{lname}-{idx}.xml")

            sname = f"data/handwriting-dataset/{data_subset}/transcriptions/{name}-{idx}.csv"

            if data_subset == "train":
                transcriptions.append(lines[i-1])
                save_names.append(sname)
            else:
                with open(sname, "w") as f:
                    f.write(",".join(encode_transcription(corpus, lines[i-1])))

                save_names.append(re.sub("transcriptions", "strokes", sname))

    if data_subset == "train":
        corpus = "".join(transcriptions)
        corpus = list(set(corpus))
        corpus = [c for c in corpus if re.match("^[a-zA-Z\.\!\?, ]$", c)]
        corpus.sort()
        corpus = dict(zip(corpus, range(len(corpus))))

        transcriptions = [encode_transcription(corpus, t) for t in transcriptions]

        for i in range(len(save_names)):
            with open(save_names[i], "w") as f:
                f.write(",".join(transcriptions[i]))

            save_names[i] = re.sub("transcriptions", "strokes", save_names[i])

    
    return load_names, save_names, corpus

def parse_xml(xml_file):
    strokes = []

    root = ET.fromstring(xml_file)
    lines = root.find("StrokeSet").findall("Stroke")

    for line in lines:
        
        strokes += [[int(point.attrib["x"]), int(point.attrib["y"]), 0] for point in line]
        strokes[-1][-1] = 1

    strokes = [[strokes[i][0] - strokes[i-1][0], strokes[i][1] - strokes[i-1][1], strokes[i][2]]\
                for i in range(1, len(strokes))]

    return strokes

def prepare_strokes(files_in):
    strokes = [None] * len(files_in)
    for i in tqdm(range(len(files_in)), ncols=c.tqdm_ncols):
        
        with open(files_in[i]) as f:
            data = f.read()
        
        strokes[i] = np.array(parse_xml(data), dtype=np.float32)

    return strokes

def standarize_and_save(file_names, data, means, stds):

    for fname, d in tqdm(zip(file_names, data), ncols=c.tqdm_ncols, total=len(file_names)):

        for i in range(2):
            d[:, i] = (d[:, i] - means[i])/stds[i]

        np.savetxt(fname, d, delimiter=",")


if __name__ == "__main__":

    dataset_dir = "data/handwriting-dataset"

    # # create directories to store the dataset
    print("Creating directories...")

    os.makedirs("handwriting-synthesis/model", exist_ok=True)

    for data_subset in ["train", "dev"]:
        for data_type in ["strokes", "transcriptions"]:
            os.makedirs(f"{dataset_dir}/{data_subset}/{data_type}", exist_ok=True)

    os.makedirs(f"{dataset_dir}/data_split", exist_ok=True)

    print("Directories ready")

    # # load file names
    print("Loading files names...")
    train_original_names = []
    dev_original_names = []

    for file in ["trainset.txt", "testset_v.txt", "testset_t.txt"]:
        train_original_names += readlines("data/" + file)

    dev_original_names = readlines("data/testset_f.txt")

    print("Files names loaded")

    # # load transcriptions and clean files names
    print("Processing transcriptions...")

    train_load_names, train_save_names, corpus = process_transcriptions_and_filenames(train_original_names, "train")
    dev_load_names, dev_save_names, _ = process_transcriptions_and_filenames(dev_original_names, "dev", corpus)

    for data_subset, fnames in zip(["train", "dev"], [train_save_names, dev_save_names]):

        fnames_cleaned = [re.sub("\.csv$", "", name.split("/")[-1]) for name in fnames]

        with open(f"{dataset_dir}/data_split/{data_subset}.txt", "w") as f:
            f.write("\n".join(fnames_cleaned))

    with open("handwriting-synthesis/model/corpus.pickle", "wb") as f:
        pickle.dump(corpus, f)

    inverse_corpus = dict(zip(corpus.values(), corpus.keys()))
    with open("handwriting-synthesis/model/inverse_corpus.pickle", "wb") as f:
        pickle.dump(inverse_corpus, f)

    print("Transcriptions ready")

    # # create preprocessed dataset
    print("XML parsing...")

    train_data = prepare_strokes(train_load_names)
    dev_data = prepare_strokes(dev_load_names)

    print("XML parsing ready")

    # # standarize
    print("Estimating standarization params...")

    tmp = np.vstack([x[:, 0:2] for x in train_data])
    means = np.mean(tmp, axis=0)
    stds = np.std(tmp, axis=0)
    np.savetxt("handwriting-synthesis/model/standarization.csv", np.vstack((means, stds)).T, delimiter=",")
    
    print("Standarization params ready")
    

    print("Standarizing and saving data...")

    standarize_and_save(train_save_names, train_data, means, stds)
    standarize_and_save(dev_save_names, dev_data, means, stds)

    print("Standarized data saved")

    print("Dataset ready")



    
