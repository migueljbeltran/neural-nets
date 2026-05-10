'''
CNN method for stage 3 image classification tasks.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.base_class.method import method


def _detect_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN(method, nn.Module):
    def __init__(self, mName=None, mDescription=None, *,
                 conv_channels=(32, 64),
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 pool='max',
                 pool_kernel=2,
                 hidden_dim=128,
                 dropout=0.3,
                 learning_rate=0.001,
                 max_epoch=8,
                 batch_size=128,
                 optimizer_name='adam',
                 loss_name='cross_entropy'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.pool = pool
        self.pool_kernel = pool_kernel
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.device = _detect_device()
        self.train_loss_history = []
        self.train_acc_history = []
        self.feature_extractor = None
        self.classifier = None

    def _build(self, c, h, w, num_classes):
        pool_cls = nn.AvgPool2d if self.pool == 'avg' else nn.MaxPool2d
        layers = []
        in_ch = c
        for out_ch in self.conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          padding=self.padding),
                nn.ReLU(),
                pool_cls(self.pool_kernel),
            ]
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*layers)

        with torch.no_grad():
            flat_dim = self.feature_extractor(torch.zeros(1, c, h, w)).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )
        self.to(self.device)

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

    def _make_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _make_loss(self):
        if self.loss_name == 'mse':
            def mse_loss(outputs, y):
                target = nn.functional.one_hot(y, num_classes=outputs.shape[1]).float()
                return nn.functional.mse_loss(torch.softmax(outputs, dim=1), target)
            return mse_loss
        if self.loss_name == 'nll':
            return lambda outputs, y: nn.functional.nll_loss(torch.log_softmax(outputs, dim=1), y)
        return nn.CrossEntropyLoss()

    def _train_loop(self, X, y):
        self.train()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=self.batch_size, shuffle=True,
        )
        optimizer = self._make_optimizer()
        loss_fn = self._make_loss()
        self.train_loss_history = []
        self.train_acc_history = []

        for epoch in range(self.max_epoch):
            total_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
                correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)
            self.train_loss_history.append(total_loss / total)
            self.train_acc_history.append(correct / total)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:>2}/{self.max_epoch} '
                      f'Loss: {self.train_loss_history[-1]:.4f} '
                      f'Acc: {self.train_acc_history[-1]:.4f}')

    def _predict(self, X):
        self.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=self.batch_size)
        preds = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                preds.append(self(X_batch).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def run(self, X_train, y_train, X_test):
        c, h, w = X_train.shape[1:]
        num_classes = int(y_train.max()) + 1
        self._build(c, h, w, num_classes)
        print(f'--training on {self.device}')
        self._train_loop(X_train, y_train)
        return self._predict(X_test)
