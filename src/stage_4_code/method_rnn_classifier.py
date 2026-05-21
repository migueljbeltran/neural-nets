'''
Small RNN classifier for Stage 4 sentiment classification.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.base_class.method import method


def detect_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_RNN_Classifier(method, nn.Module):
    def __init__(self, mName=None, mDescription=None, *,
                 vocab_size=10000,
                 cell_type='rnn',
                 embedding_dim=64,
                 hidden_dim=128,
                 dropout=0.4,
                 learning_rate=0.001,
                 max_epoch=4,
                 batch_size=128):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = detect_device()
        self.train_loss_history = []
        self.train_acc_history = []

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_class = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type]
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.to(self.device)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_indices = (lengths - 1).view(-1, 1, 1)
        last_indices = last_indices.expand(-1, 1, self.hidden_dim)
        last_output = output.gather(1, last_indices).squeeze(1)
        return self.classifier(last_output)

    def train_model(self, X, lengths, y):
        self.train()
        loader = DataLoader(
            TensorDataset(
                torch.LongTensor(X),
                torch.LongTensor(lengths),
                torch.LongTensor(y),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.train_loss_history = []
        self.train_acc_history = []

        for epoch in range(self.max_epoch):
            total_loss, correct, total = 0.0, 0, 0
            for X_batch, length_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                length_batch = length_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.forward(X_batch, length_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)

            self.train_loss_history.append(total_loss / total)
            self.train_acc_history.append(correct / total)
            print(f'  Epoch {epoch + 1:>2}/{self.max_epoch} '
                  f'Loss: {self.train_loss_history[-1]:.4f} '
                  f'Acc: {self.train_acc_history[-1]:.4f}')

    def test_model(self, X, lengths):
        self.eval()
        loader = DataLoader(
            TensorDataset(torch.LongTensor(X), torch.LongTensor(lengths)),
            batch_size=self.batch_size,
        )
        predictions = []
        with torch.no_grad():
            for X_batch, length_batch in loader:
                X_batch = X_batch.to(self.device)
                length_batch = length_batch.to(self.device)
                outputs = self.forward(X_batch, length_batch)
                predictions.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions)

    def run(self, X_train, train_lengths, y_train, X_test, test_lengths):
        print(f'--training {self.cell_type.upper()} classifier on {self.device}')
        self.train_model(X_train, train_lengths, y_train)
        print('--testing classifier')
        return self.test_model(X_test, test_lengths)
