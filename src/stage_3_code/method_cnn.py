'''
CNN method for stage 3 image classification tasks.
'''

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.base_class.method import method


class Method_CNN(method, nn.Module):
    learning_rate = 0.001
    max_epoch = 8
    batch_size = 128
    optimizer_name = 'adam'

    def __init__(self, mName=None, mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv_channels = [32, 64]
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.pool_kernel = 2
        self.hidden_dim = 128
        self.dropout = 0.3
        self.loss_name = 'cross_entropy'
        self.model = None
        self.loss_function = None
        self.train_loss_history = []
        self.train_acc_history = []

    def _build_network(self, input_shape, num_classes):
        c, h, w = input_shape
        layers = []
        in_ch = c

        for out_ch in self.conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=self.pool_kernel))
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        with torch.no_grad():
            sample = torch.zeros((1, c, h, w))
            flat_dim = self.feature_extractor(sample).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )
        self.model = nn.Sequential(self.feature_extractor, self.classifier).to(self.device)

    def _make_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _compute_loss(self, outputs, y_batch):
        if self.loss_name == 'nll':
            return nn.NLLLoss()(torch.log_softmax(outputs, dim=1), y_batch)
        if self.loss_name == 'mse':
            y_one_hot = nn.functional.one_hot(y_batch, num_classes=outputs.shape[1]).float()
            return nn.MSELoss()(torch.softmax(outputs, dim=1), y_one_hot)
        return self.loss_function(outputs, y_batch)

    def train_model(self, X, y):
        self.train()
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = self._make_optimizer()

        self.train_loss_history = []
        self.train_acc_history = []

        for epoch in range(self.max_epoch):
            epoch_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self._compute_loss(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * X_batch.size(0)
                correct += int((outputs.argmax(dim=1) == y_batch).sum().item())
                total += int(X_batch.size(0))

            avg_loss = epoch_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(avg_acc)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f'  Epoch {epoch + 1:>2}/{self.max_epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}')

    def test_model(self, X):
        self.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            return outputs.argmax(dim=1).cpu().numpy()

    def run(self, X_train, y_train, X_test):
        input_shape = tuple(X_train.shape[1:])
        num_classes = int(y_train.max()) + 1
        self._build_network(input_shape, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        print(f'--training on device: {self.device}')
        self.train_model(X_train, y_train)
        return self.test_model(X_test)

    def clone_with_updates(self, **kwargs):
        copied = copy.deepcopy(self)
        for k, v in kwargs.items():
            setattr(copied, k, v)
        return copied
