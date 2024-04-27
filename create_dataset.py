import xml.etree.ElementTree as ET
import numpy as np
import config as c
import tarfile
import io
import re
from tqdm import tqdm
from dataset import load_data_from_tarfile

# this code operates on tar.gz files of IAM-onDB;
# operations on archives of that scale are slow;
# this code was meant more for better understanding of tarfile api than for speed

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

def save2archive(archive, data, name, **kwargs):
    buffer = io.BytesIO()
    np.savetxt(buffer, data, delimiter=",", **kwargs)
    tarinfo = tarfile.TarInfo(name=name)
    tarinfo.size = buffer.getbuffer().nbytes
    buffer.seek(0)
    archive.addfile(tarinfo, buffer)

def prepare_data(archive_in_name, archive_out_name):
    print("XML parsing...")

    archive_in = tarfile.open(archive_in_name)
    archive_out = tarfile.open(archive_out_name, "w:gz")
    for member in tqdm(archive_in.getmembers(), ncols=c.tqdm_ncols):
        if not ".xml" in member.name:
            continue
        
        f = archive_in.extractfile(member.name)
        data = parse_xml(f.read())

        save2archive(archive_out, data, re.sub("\.xml$", ".csv", member.name), fmt="%.0f")

    archive_in.close()
    archive_out.close()

    print("XML parsing ready")

def standarize_and_save(file_names, data, archive_name, mean, std):

    with tarfile.open(archive_name, "w:gz") as archive:
        for fname, d in tqdm(zip(file_names, data), ncols=c.tqdm_ncols):
            d[:, 0:2] = (d[:, 0:2] - mean)/std

            save2archive(archive, d, fname)


if __name__ == "__main__":
    print("Preparing dataset...")

    # # create preprocessed dataset
    prepare_data("data/lineStrokes-all.tar.gz", "data/online_handwriting.tar.gz")

    # # create train/test split
    with tarfile.open("data/online_handwriting.tar.gz") as archive:
        files = [m.name for m in archive.getmembers() if ".csv" in m.name]

    idxs = list(range(len(files)))
    np.random.shuffle(idxs)
    div = int(len(idxs) * c.test_frac)
    with open("data_split/dev.txt", "w") as f:
        f.write("\n".join([files[idxs[i]] for i in range(div)]))

    with open("data_split/train.txt", "w") as f:
        f.write("\n".join([files[idxs[i]] for i in range(div+1, len(files))]))

    # # estimate offsets standarization params
    print("Estimating standarization params...")
    dev_names, dev_data = load_data_from_tarfile("data_split/dev.txt", "data/online_handwriting.tar.gz")
    train_names, train_data = load_data_from_tarfile("data_split/train.txt", "data/online_handwriting.tar.gz")

    tmp = np.vstack([x[:, 0:2] for x in train_data])
    mean = np.mean(tmp)
    std_dev = np.std(tmp)
    with open("standarization.txt", "w") as f:
        f.write("Mean,Std\n" + str(mean) + "," + str(std_dev))
    
    print("Standarization params ready")
    
    # # unpack before training for faster loading time
    print("Data standarizing and saving...")
    standarize_and_save(dev_names, dev_data, "data/online_handwriting_dev.tar.gz", mean, std_dev)
    standarize_and_save(train_names, train_data, "data/online_handwriting_train.tar.gz", mean, std_dev)
    print("Data standarized and saved")

    print("Dataset ready")



    
