'''
Stage 4 RNN experiments for text classification and generation.

Run from the repo root:
    python3 -m script.stage_4_script.script_rnn
'''

import os
import random

import numpy as np
import torch

from src.stage_4_code.dataset_loader import Dataset_Loader
from src.stage_4_code.evaluate_metrics import Evaluate_Metrics
from src.stage_4_code.method_rnn_classifier import Method_RNN_Classifier
from src.stage_4_code.method_rnn_generator import Method_RNN_Generator
from src.stage_4_code.result_saver import Result_Saver


SEED = 2
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'stage_4_data')
RESULT_DIR = os.path.join(REPO_ROOT, 'result', 'stage_4_result')

CLASSIFICATION_DIR = os.path.join(DATA_DIR, 'text_classification')
GENERATION_DIR = os.path.join(DATA_DIR, 'text_generation')

CLASSIFICATION_MAX_VOCAB = 10000
CLASSIFICATION_MAX_LENGTH = 200
GENERATION_MAX_VOCAB = 3000
GENERATION_SEQUENCE_LENGTH = 5
SEED_WORDS = 'what did the'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_history(name, loss_history, acc_history):
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f'{name}_history.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,loss,accuracy\n')
        for i, (loss, acc) in enumerate(zip(loss_history, acc_history), start=1):
            f.write(f'{i},{loss:.6f},{acc:.6f}\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'matplotlib not installed, saved {csv_path}')
        return

    epochs = np.arange(1, len(loss_history) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, loss_history, 'b-o', markersize=3)
    ax1.set(title='Training Loss', xlabel='Epoch', ylabel='Loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, np.array(acc_history) * 100.0, 'g-o', markersize=3)
    ax2.set(title='Training Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    fig.suptitle(name.replace('_', ' ').title())
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f'{name}_learning_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_pickle(name, data):
    saver = Result_Saver('saver', '')
    saver.result_destination_folder_path = RESULT_DIR
    saver.result_destination_file_name = name
    saver.data = data
    saver.save()


def run_classification():
    print('\n===== Text Classification =====')
    data_obj = Dataset_Loader(
        'IMDb Reviews',
        '',
        task='classification',
        max_vocab_size=CLASSIFICATION_MAX_VOCAB,
        max_length=CLASSIFICATION_MAX_LENGTH,
    )
    data_obj.dataset_source_folder_path = CLASSIFICATION_DIR
    data = data_obj.load()

    X_train = data['train']['X']
    train_lengths = data['train']['lengths']
    y_train = data['train']['y']
    X_test = data['test']['X']
    test_lengths = data['test']['lengths']
    y_test = data['test']['y']

    scores = {}
    evaluator = Evaluate_Metrics('metrics', '')
    for cell_type in ['rnn', 'lstm', 'gru']:
        set_seed()
        model = Method_RNN_Classifier(
            f'{cell_type}-classifier',
            '',
            vocab_size=len(data['vocab']),
            cell_type=cell_type,
            hidden_dim=128,
            dropout=0.4,
            max_epoch=4,
        )
        pred_y = model.run(X_train, train_lengths, y_train, X_test, test_lengths)
        evaluator.data = {'true_y': y_test, 'pred_y': pred_y}
        score = evaluator.evaluate()
        scores[cell_type] = score
        save_pickle(f'{cell_type}_classification_result.pkl', {
            'pred_y': pred_y,
            'true_y': y_test,
            'scores': score,
            'train_loss_history': model.train_loss_history,
            'train_acc_history': model.train_acc_history,
        })
        save_history(f'{cell_type}_classification',
                     model.train_loss_history,
                     model.train_acc_history)
        print(f'{cell_type.upper()} scores: {score}')
    return scores


def run_generation():
    print('\n===== Text Generation =====')
    data_obj = Dataset_Loader(
        'Jokes',
        '',
        task='generation',
        max_vocab_size=GENERATION_MAX_VOCAB,
    )
    data_obj.dataset_source_folder_path = GENERATION_DIR
    data_obj.dataset_source_file_name = 'data'
    data = data_obj.load()

    generations = {}
    for cell_type in ['rnn', 'lstm', 'gru']:
        set_seed()
        model = Method_RNN_Generator(
            f'{cell_type}-generator',
            '',
            vocab_size=len(data['vocab']),
            cell_type=cell_type,
            sequence_length=GENERATION_SEQUENCE_LENGTH,
            max_epoch=10,
        )
        text = model.run(
            data['texts'],
            data['vocab'],
            data['id_to_word'],
            seed_words=SEED_WORDS,
        )
        generations[cell_type] = text
        save_pickle(f'{cell_type}_generation_result.pkl', {
            'generated_text': text,
            'seed_words': SEED_WORDS,
            'train_loss_history': model.train_loss_history,
            'train_acc_history': model.train_acc_history,
        })
        save_history(f'{cell_type}_generation',
                     model.train_loss_history,
                     model.train_acc_history)
        print(f'{cell_type.upper()} generated text: {text}')
    return generations


def write_summary(scores, generations):
    path = os.path.join(RESULT_DIR, 'summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Stage 4 Summary\n\n')
        f.write('Classification\n')
        for name, score in scores.items():
            f.write(f'{name.upper()}: {score}\n')
        f.write('\nGeneration\n')
        for name, text in generations.items():
            f.write(f'{name.upper()}: {text}\n')
    print(f'\nsummary saved to {path}')


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    set_seed()
    scores = run_classification()
    generations = run_generation()
    write_summary(scores, generations)


if __name__ == '__main__':
    main()
