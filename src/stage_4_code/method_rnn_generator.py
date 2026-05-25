'''
Small RNN text generator for Stage 4.
'''

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.base_class.method import method
from src.stage_4_code.dataset_loader import UNK_TOKEN, tokenize
from src.stage_4_code.method_rnn_classifier import detect_device


class Method_RNN_Generator(method, nn.Module):
    def __init__(self, mName=None, mDescription=None, *,
                 vocab_size=3000,
                 cell_type='rnn',
                 sequence_length=5,
                 embedding_dim=64,
                 hidden_dim=64,
                 learning_rate=0.001,
                 max_epoch=10,
                 batch_size=128):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.device = detect_device()
        self.train_loss_history = []
        self.train_acc_history = []

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        rnn_layers = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        if cell_type not in rnn_layers:
            raise ValueError('cell_type must be one of: rnn, lstm, gru')
        rnn_class = rnn_layers[cell_type]
        self.rnn = rnn_class(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.to(self.device)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        if self.cell_type == 'lstm':
            hidden = hidden[0]
        return self.output(hidden[-1])

    def _make_training_pairs(self, texts, vocab):
        X, y = [], []
        unk_id = vocab[UNK_TOKEN]
        for text in texts:
            ids = [vocab.get(word, unk_id) for word in tokenize(text)]
            for i in range(len(ids) - self.sequence_length):
                X.append(ids[i:i + self.sequence_length])
                y.append(ids[i + self.sequence_length])
        return torch.LongTensor(X), torch.LongTensor(y)

    def train_model(self, texts, vocab):
        self.train()
        X, y = self._make_training_pairs(texts, vocab)
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.train_loss_history = []
        self.train_acc_history = []

        for epoch in range(self.max_epoch):
            total_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
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

    def generate(self, seed_words, vocab, id_to_word, max_words=18, temperature=0.7):
        self.eval()
        words = tokenize(seed_words)
        if not words:
            words = ['what', 'did', 'the']
        unk_id = vocab[UNK_TOKEN]

        with torch.no_grad():
            for _ in range(max_words):
                ids = [vocab.get(word, unk_id) for word in words[-self.sequence_length:]]
                if len(ids) < self.sequence_length:
                    ids = [0] * (self.sequence_length - len(ids)) + ids
                x = torch.LongTensor([ids]).to(self.device)
                logits = self.forward(x)[0] / temperature
                logits[0] = -1e9
                logits[unk_id] = -1e9
                probs = torch.softmax(logits, dim=0)
                next_id = int(torch.multinomial(probs, 1).item())
                words.append(id_to_word.get(next_id, UNK_TOKEN))
        return ' '.join(words)

    def run(self, texts, vocab, id_to_word, seed_words='what did the'):
        random.seed(2)
        torch.manual_seed(2)
        print(f'--training {self.cell_type.upper()} generator on {self.device}')
        self.train_model(texts, vocab)
        print('--generating text')
        return self.generate(seed_words, vocab, id_to_word)
