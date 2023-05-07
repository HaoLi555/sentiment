import torch
import torch.nn as nn
from utils import load_index2vec

I2V_PATH = "Preprocess/index2vec.json"
EMBEDDING_DIM = 50
KERNAL_HEIGH = [3, 5, 7]


class baseline(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        index2vec = load_index2vec(I2V_PATH)
        self.embedding = nn.Embedding(len(index2vec), EMBEDDING_DIM)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.tensor(index2vec))

        self.hidden = nn.Sequential(
            nn.Linear(args.seq_length * EMBEDDING_DIM, 128),
            nn.Dropout(p=args.drop_out),
            nn.Linear(128, 256),
            nn.Dropout(p=args.drop_out),
            nn.Linear(256, 512),
            nn.Dropout(p=args.drop_out),
            nn.Linear(512, 256),
            nn.Dropout(p=args.drop_out),
            nn.Linear(256, 128),
            nn.Dropout(p=args.drop_out),
            nn.Linear(128, 2),
            nn.Dropout(p=args.drop_out),
        )

        self.out = nn.Sequential(nn.LogSoftmax(-1))

    def forward(self, x):
        x = self.embedding(x.type(torch.long))

        x = torch.reshape(x, (x.size(0), -1))  # flatten
        x_hid = self.hidden(x)
        y = self.out(x_hid)
        return y


class CNN(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        index2vec = load_index2vec(I2V_PATH)
        self.embedding = nn.Embedding(len(index2vec), EMBEDDING_DIM)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.tensor(index2vec))

        # input shape (batch_size, 1, seq_len, EMBEDDING_DIM)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(KERNAL_HEIGH[0], EMBEDDING_DIM),
                stride=1,
            ),
            nn.MaxPool2d(kernel_size=(args.seq_length - KERNAL_HEIGH[0] + 1, 1)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(KERNAL_HEIGH[1], EMBEDDING_DIM),
                stride=1,
            ),
            nn.MaxPool2d(kernel_size=(args.seq_length - KERNAL_HEIGH[1] + 1, 1)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(KERNAL_HEIGH[2], EMBEDDING_DIM),
                stride=1,
            ),
            nn.MaxPool2d(kernel_size=(args.seq_length - KERNAL_HEIGH[2] + 1, 1)),
        )

        self.drop_out = nn.Dropout(p=args.drop_out)

        # shape now (bs, 3*out_channel, 1, 1)
        self.out = nn.Sequential(nn.Linear(12 * 1 * 1, 2), nn.LogSoftmax(-1))
        # shape now (bs, 2)

    def forward(self, x):
        # print(f"x.shape :{x.shape}")
        x = self.embedding(x.type(torch.long))
        # print(f"embedded shape: {x.shape}")
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv = torch.cat((conv1, conv2, conv3), dim=1)
        # print(f"conv1.shape: {conv1.shape}")
        # print(f"conv2.shape: {conv2.shape}")
        # print(f"conv3.shape: {conv3.shape}")
        # print(f"conv.shape: {conv.shape}")
        conv = conv.view(conv.size(0), -1)  # flatten the output to (batch_size,12*1*1)
        y = self.out(self.drop_out(conv))
        return y


class RNN(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        index2vec = load_index2vec(I2V_PATH)
        self.embedding = nn.Embedding(len(index2vec), EMBEDDING_DIM)
        self.embedding.weight.requires_grad = True
        self.embedding.weight.data.copy_(torch.tensor(index2vec))

        self.rnn = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=100,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
        )

        self.drop_out = nn.Dropout(p=args.drop_out)

        # in: seq*hidden_size*num_directions
        self.out = nn.Sequential(
            nn.Linear(args.seq_length * 100 * 2, 2), nn.LogSoftmax(-1)
        )

    def forward(self, x):
        x = self.embedding(x.type(torch.long))
        y, (h_t, c_t) = self.rnn(x)
        y = self.drop_out(y)
        y = torch.reshape(y, (y.size(0), -1))  # flatten
        y = self.out(y)
        return y
