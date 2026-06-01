'''
Two-layer Graph Convolutional Network for Stage 5 node classification.

Implements the spectral graph convolution from Kipf & Welling (2017),
"Semi-Supervised Classification with Graph Convolutional Networks".
The normalized adjacency (with self-loops) is precomputed by the dataset
loader and passed in as a torch sparse tensor.
'''

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base_class.method import method


def detect_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class GraphConvolution(nn.Module):
    '''Single graph convolution layer: A_hat @ X @ W (+ b).'''

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot / Xavier uniform, matching the reference GCN implementation
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class Method_GCN(method, nn.Module):
    def __init__(self, mName=None, mDescription=None, *,
                 num_features,
                 num_classes,
                 hidden_dim=16,
                 num_layers=2,
                 dropout=0.5,
                 learning_rate=0.01,
                 weight_decay=5e-4,
                 max_epoch=200):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.verbose = True
        self.device = detect_device()

        # Build a stack of `num_layers` graph-convolution layers.
        # 1 layer  -> features -> classes (no hidden layer)
        # 2 layers -> features -> hidden -> classes (standard GCN)
        # k layers -> features -> hidden -> ... -> hidden -> classes
        dims = [num_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList(
            GraphConvolution(dims[i], dims[i + 1]) for i in range(num_layers))

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        self.to(self.device)

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def accuracy(output, labels):
        preds = output.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)

    def train_model(self, features, adj, labels, idx_train, idx_val):
        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_val = idx_val.to(self.device)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        for epoch in range(self.max_epoch):
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = self.accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                output = self.forward(features, adj)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = self.accuracy(output[idx_val], labels[idx_val])

            self.train_loss_history.append(loss_train.item())
            self.train_acc_history.append(acc_train)
            self.val_loss_history.append(loss_val.item())
            self.val_acc_history.append(acc_val)

            if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                print(f'  Epoch {epoch + 1:>3}/{self.max_epoch} '
                      f'train_loss: {loss_train.item():.4f} '
                      f'train_acc: {acc_train:.4f} '
                      f'val_loss: {loss_val.item():.4f} '
                      f'val_acc: {acc_val:.4f}')

    def test_model(self, features, adj, idx_test):
        features = features.to(self.device)
        adj = adj.to(self.device)
        idx_test = idx_test.to(self.device)
        self.eval()
        with torch.no_grad():
            output = self.forward(features, adj)
            preds = output[idx_test].argmax(dim=1)
        return preds.cpu().numpy()

    def run(self, features, adj, labels, idx_train, idx_val, idx_test):
        if self.verbose:
            print(f'--training GCN on {self.device}')
        self.train_model(features, adj, labels, idx_train, idx_val)
        if self.verbose:
            print('--testing GCN')
        return self.test_model(features, adj, idx_test)
