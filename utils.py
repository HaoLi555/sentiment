import os
import json
import gensim
import numpy as np

SAVE_PATH_W2I = "Preprocess/word2index.json"
SAVE_PATH_I2V = "Preprocess/index2vec.json"
DATA_SET_PATH = "Dataset"
PRETRAINED_W2V_PATH = "Pretrained/wiki_word2vec_50.bin"

def generate_word2index(save_path, data_path):
    """generate a dict giving every word an index

    Args:
        save_path (str): save path of word2index(dict)
        data_path (str): path of data set DIR which contains all the data sets to be used

    Returns:
        dict: find index of a word
    """
    word2index = {}
    files = os.listdir(data_path)
    # create dict
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.strip("\n").split()
                for word in line[1:]:  # the first one is label
                    if word not in word2index.keys():
                        word2index[word] = len(word2index) + 1
    # save
    with open(save_path, "w") as savef:
        json.dump(word2index, savef, ensure_ascii=False)

    return word2index


def generate_index2vec(save_path, data_path, word2index: dict):
    """generate a ndarray finding word vector by index

    Args:
        save_path (str): save path of index2vec(list)
        data_path (str): path of pretrained word vector
        word2index (dict): find index of word

    Returns:
        ndarray: find a vector by index
    """

    index2vec = np.random.random(size=(len(word2index) + 1, 50))

    # create
    print("generating index2vec...")
    model = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=True)
    for word in word2index.keys():
        try:
            index2vec[word2index[word]] = model.get_vector(word)
        except:
            print(
                f"error while generating index2vec: maybe because key({word}) not found in w2v"
            )
            index2vec[word2index[word]] = np.random.randn(50)
    index2vec[0] = np.zeros(50)
    print("finish generating index2vec ")

    # save
    with open(save_path, "w") as savef:
        json.dump(index2vec.tolist(), savef)

    return index2vec


def load_word2index(path):
    """load word2index

    Args:
        path (str): path of word2index.json

    Returns:
        dict: word2index
    """
    word2index = {}
    with open(path, "r") as f:
        word2index = json.loads(f.read())
    return word2index


def load_index2vec(path):
    """load index2vec

    Args:
        path (str): path of index2vec.json

    Returns:
        list: index2vec
    """

    index2vec = []
    with open(path, "r") as f:
        index2vec = json.loads(f.read())
    return index2vec


if __name__ == "__main__":
    generate_index2vec(
        SAVE_PATH_I2V,
        PRETRAINED_W2V_PATH,
        generate_word2index(SAVE_PATH_W2I, DATA_SET_PATH),
    )
