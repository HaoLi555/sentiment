import torch
import torch.nn as nn
from model import RNN, CNN, baseline
from dataloader import get_dataloaders
from tqdm import *
from sklearn.metrics import f1_score


DATA_PATH = "Dataset"
W2I_PATH = "Preprocess/word2index.json"


class NNManager:
    def __init__(self, args) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        if args.model == "CNN":
            cnn = CNN(args=args)
            if not args.train:
                cnn.load_state_dict(torch.load(args.load_cnn_model_path))
            self.model = cnn
        elif args.model == "RNN":
            rnn = RNN(args=args)
            if not args.train:
                rnn.load_state_dict(torch.load(args.load_rnn_model_path))
            self.model = rnn
        elif args.model == "baseline":
            bl = baseline(args=args)
            if not args.train:
                bl.load_state_dict(torch.load(args.load_baseline_model_path))
            self.model = bl

        self.model.to(device=self.device)
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.save = args.save
        self.train = args.train

        data_loaders = get_dataloaders(
            DATA_PATH,
            W2I_PATH,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            train=self.train,
        )
        self.val_loader = data_loaders["validation"]
        self.test_loader = data_loaders["test"]
        if self.train:
            self.train_loader = data_loaders["train"]

        self.loss_func = nn.CrossEntropyLoss().to(device=self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=args.weight_decay,
        )

    def _train(self):
        for epoch in trange(self.epochs):
            self.model.train()
            for step, (x, label) in enumerate(tqdm(self.train_loader)):
                x = x.to(self.device)
                label = label.to(self.device)

                input = []
                if self.args.model == "CNN":
                    input = x.unsqueeze(
                        1
                    )  # [bs, seq_length] -> [bs, 1(channel), seq_length]
                elif self.args.model == "RNN":
                    input = x
                elif self.args.model == "baseline":
                    input = x

                y_pred = self.model(input)

                # print(y_pred.shape)
                # print(label.shape)

                loss = self.loss_func(y_pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"***** Epoch: {epoch}: Eval results *****")
            self._print_results("train")
            if epoch % 3 == 0:
                val_outputs = self._get_outputs("validate")
                print(f"  validate_loss: {val_outputs['loss']}")
                print(f"  acc: {val_outputs['acc']}")
                print(f"  f1_score: {val_outputs['f1_score']}")
                if val_outputs["f1_score"] > 0.85:
                    print("early stop!")
                    break

    def _save(self):
        save_path = ""
        if self.args.model == "CNN":
            save_path = self.args.save_cnn_model_path
        elif self.args.model == "RNN":
            save_path = self.args.save_rnn_model_path
        elif self.args.model == "baseline":
            save_path = self.args.save_baseline_model_path

        torch.save(self.model.state_dict(), save_path)

    def _get_outputs(self, data_set):
        """get outputs of train, validation or test

        Args:
            data_set (str): options: {'train', 'test', 'validate'}

        Returns:
            dict: outputs with keys: 'loss', 'f1_score', 'acc'
        """

        self.model.eval()
        loss = 0.0
        acc = 0.0
        _f1_score = 0.0
        correct = 0
        y_pred_tot: list = []
        label_tot: list = []
        if data_set == "train":
            data_loader = self.train_loader
        elif data_set == "test":
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        tot = len(data_loader.dataset)

        for step, (x, label) in enumerate(data_loader):
            x = x.to(self.device)
            label = label.to(self.device)

            input = []
            if self.args.model == "CNN":
                input = x.unsqueeze(
                    1
                )  # [bs, seq_length] -> [bs, 1(channel), seq_length]
            elif self.args.model == "RNN":
                input = x
            elif self.args.model == "baseline":
                input = x
            y_out = self.model(input)  # probabilty
            loss += self.loss_func(y_out, label).item()
            y_pred = torch.argmax(input=y_out, dim=1)
            correct += (y_pred == label).sum().item()
            y_pred_tot.extend(y_pred)
            label_tot.extend(label)

        loss /= tot / self.batch_size
        acc = float(correct) / tot
        _f1_score = f1_score(
            torch.tensor(y_pred_tot).to(torch.device("cpu")),
            torch.tensor(label_tot).to(torch.device("cpu")),
        )

        outputs = {"loss": loss, "acc": acc, "f1_score": _f1_score}

        return outputs

    def _print_results(self, data_set):
        """print results

        Args:
            data_set (str): options: {'train', 'test', 'validate'}
        """
        if data_set == "train":
            outputs = self._get_outputs("train")
            print(f"  train_loss: {outputs['loss']}")
            print(f"  acc: {outputs['acc']}")
            print(f"  f1_score: {outputs['f1_score']}")
        else:
            if data_set == "test":
                outputs = self._get_outputs("test")
            else:
                outputs = self._get_outputs("validate")
            print(f"  loss: {outputs['loss']}")
            print(f"  acc: {outputs['acc']}")
            print(f"  f1_score: {outputs['f1_score']}")
