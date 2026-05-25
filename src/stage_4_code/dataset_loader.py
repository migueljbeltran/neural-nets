'''
Dataset loader for Stage 4 text classification and generation.
'''

import csv
import os
import re
from collections import Counter

import numpy as np

from src.base_class.dataset import dataset


PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(texts, max_vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def encode_text(text, vocab, max_length):
    ids = [vocab.get(word, vocab[UNK_TOKEN]) for word in tokenize(text)]
    length = min(len(ids), max_length)
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids += [vocab[PAD_TOKEN]] * (max_length - len(ids))
    return ids, max(length, 1)


class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None, *,
                 task='classification',
                 max_vocab_size=10000,
                 max_length=200):
        super().__init__(dName, dDescription)
        self.task = task
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length

    def _read_review_split(self, root, split):
        texts, labels = [], []
        for label_name, label_id in [('neg', 0), ('pos', 1)]:
            folder = os.path.join(root, split, label_name)
            for file_name in sorted(os.listdir(folder)):
                if not file_name.endswith('.txt'):
                    continue
                path = os.path.join(folder, file_name)
                with open(path, encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                labels.append(label_id)
        return texts, labels

    def _load_classification(self):
        root = self.dataset_source_folder_path
        train_texts, train_y = self._read_review_split(root, 'train')
        test_texts, test_y = self._read_review_split(root, 'test')
        vocab = build_vocab(train_texts, self.max_vocab_size)

        train_encoded = [encode_text(text, vocab, self.max_length) for text in train_texts]
        test_encoded = [encode_text(text, vocab, self.max_length) for text in test_texts]
        X_train, train_lengths = zip(*train_encoded)
        X_test, test_lengths = zip(*test_encoded)
        X_train = np.asarray(X_train, dtype=np.int64)
        X_test = np.asarray(X_test, dtype=np.int64)
        train_lengths = np.asarray(train_lengths, dtype=np.int64)
        test_lengths = np.asarray(test_lengths, dtype=np.int64)

        self.data = {
            'train': {
                'X': X_train,
                'lengths': train_lengths,
                'y': np.asarray(train_y, dtype=np.int64),
            },
            'test': {
                'X': X_test,
                'lengths': test_lengths,
                'y': np.asarray(test_y, dtype=np.int64),
            },
            'vocab': vocab,
            'raw': {
                'train_texts': train_texts,
                'test_texts': test_texts,
            },
        }
        print(f'classification train shape: {X_train.shape}, test shape: {X_test.shape}')
        print(f'classification vocab size: {len(vocab)}')
        return self.data

    def _load_generation(self):
        path = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name,
        )
        texts = []
        with open(path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                joke = row.get('Joke', '').strip()
                if joke:
                    texts.append(joke)

        vocab = build_vocab(texts, self.max_vocab_size)
        id_to_word = {idx: word for word, idx in vocab.items()}
        self.data = {
            'texts': texts,
            'vocab': vocab,
            'id_to_word': id_to_word,
        }
        print(f'generation text count: {len(texts)}')
        print(f'generation vocab size: {len(vocab)}')
        return self.data

    def load(self):
        if self.task == 'generation':
            return self._load_generation()
        return self._load_classification()
