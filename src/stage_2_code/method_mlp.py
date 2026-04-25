'''
Method class for MLP on MNIST (10-class classification)
Input:  784 pixel features
Output: 10 digit classes
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.base_class.method import method


class Method_MLP(method, nn.Module):

    learning_rate = 1e-3
    max_epoch     = 20
    batch_size    = 256

    def __init__(self, mName=None, mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.optimizer_class  = torch.optim.Adam
        self.optimizer_kwargs = {}

        # Architecture: 784 -> 512 -> 256 -> 128 -> 10
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.train_loss_history = []
        self.train_acc_history  = []

    def forward(self, x):
        return self.model(x)

    def train_model(self, X, y):
        self.train()
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=self.batch_size, shuffle=True
        )

        self.train_loss_history = []
        self.train_acc_history  = []

        for epoch in range(self.max_epoch):
            epoch_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = self.loss_function(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)
                correct    += (outputs.argmax(dim=1) == y_batch).sum().item()
                total      += X_batch.size(0)

            avg_loss = epoch_loss / total
            avg_acc  = correct / total
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(avg_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1:>3}/{self.max_epoch}  Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}')

    def test_model(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(torch.FloatTensor(X))
            return outputs.argmax(dim=1).numpy()

    def run(self, X_train, y_train, X_test):
        print('--start training...')
        self.train_model(X_train, y_train)
        print('--start testing...')
        return self.test_model(X_test)
