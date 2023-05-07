import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from utils import load_word2index, load_index2vec


def get_dataloaders(data_path, word2index_path, seq_length, batch_size, train):
    """get data loaders of validation,test(and train optional) set

    Args:
        data_path (str): path of DIR of data set(train, validation, test)
        word2index_path (str): path of word2index.json
        seq_length (int): sequence length(how many words in a sequence)
        batch_size (int): batch size
        train (bool): whether to generate train data loader

    Returns:
        dict: find a data loader by 'test', 'validation' or 'train'
    """

    data_loaders = {}
    word2index = load_word2index(word2index_path)

    for file in os.listdir(data_path):
        if file == "train.txt" and not train:
            continue
        file_path = os.path.join(data_path, file)
        contents, labels = [], []
        with open(file_path, "r") as f:
            for line in f.readlines():
                words = line.strip("\n").split()
                content = []
                if len(words) >= seq_length + 1:
                    for i in range(1, seq_length + 1):
                        content.append(word2index[words[i]])
                else:
                    for i in range(1, len(words)):
                        content.append(word2index[words[i]])
                    for i in range(len(words), seq_length + 1):
                        content.append(0)  # 0 stands for nothing
                contents.append(content)
                labels.append(int(words[0]))

        labels = torch.tensor(labels)
        contents = torch.tensor(np.array(contents)).type(torch.float32)

        data_set = TensorDataset(contents, labels)
        data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
        data_loaders[file[:-4]] = data_loader

    return data_loaders
