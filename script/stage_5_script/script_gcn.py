'''
Stage 5 GCN experiments for graph node classification.

Trains a two-layer GCN on the Cora, Citeseer, and Pubmed citation networks,
classifies the held-out test nodes, saves learning curves, and reports the
evaluation metrics.

Run from the repo root:
    python3 -m script.stage_5_script.script_gcn
'''

import json
import os
import random

import numpy as np
import torch

from src.stage_5_code.dataset_loader import Dataset_Loader
from src.stage_5_code.evaluate_metrics import Evaluate_Metrics
from src.stage_5_code.method_gcn import Method_GCN
from src.stage_5_code.result_saver import Result_Saver


SEED = 2
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'stage_5_data')
RESULT_DIR = os.path.join(REPO_ROOT, 'result', 'stage_5_result')

# per-dataset training hyper-parameters (hidden=16 is the standard GCN setup)
DATASET_CONFIG = {
    'cora': {'hidden_dim': 16, 'dropout': 0.5, 'learning_rate': 0.01,
             'weight_decay': 5e-4, 'max_epoch': 200},
    'citeseer': {'hidden_dim': 16, 'dropout': 0.5, 'learning_rate': 0.01,
                 'weight_decay': 5e-4, 'max_epoch': 200},
    'pubmed': {'hidden_dim': 16, 'dropout': 0.5, 'learning_rate': 0.01,
               'weight_decay': 5e-4, 'max_epoch': 200},
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(name, history):
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f'{name}_history.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
        rows = zip(history['train_loss'], history['train_acc'],
                   history['val_loss'], history['val_acc'])
        for i, (tl, ta, vl, va) in enumerate(rows, start=1):
            f.write(f'{i},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f}\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f'matplotlib not installed, saved {csv_path}')
        return

    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history['train_loss'], 'b-', label='train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='val')
    ax1.set(title='Loss', xlabel='Epoch', ylabel='NLL Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, np.array(history['train_acc']) * 100.0, 'b-', label='train')
    ax2.plot(epochs, np.array(history['val_acc']) * 100.0, 'r-', label='val')
    ax2.set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.legend()
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


def run_dataset(dataset_name):
    print(f'\n===== {dataset_name.upper()} =====')
    set_seed()

    data_obj = Dataset_Loader(dName=dataset_name)
    data_obj.dataset_source_folder_path = os.path.join(DATA_DIR, dataset_name)
    data = data_obj.load()

    graph = data['graph']
    features = graph['X']
    adj = graph['utility']['A']
    labels = graph['y']
    splits = data['train_test_val']
    idx_train = splits['idx_train']
    idx_val = splits['idx_val']
    idx_test = splits['idx_test']

    config = DATASET_CONFIG[dataset_name]
    model = Method_GCN(
        f'{dataset_name}-gcn', '',
        num_features=data['num_features'],
        num_classes=data['num_classes'],
        **config,
    )
    pred_y = model.run(features, adj, labels, idx_train, idx_val, idx_test)
    true_y = labels[idx_test].cpu().numpy()

    evaluator = Evaluate_Metrics('metrics', '')
    evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
    score = evaluator.evaluate()

    history = {
        'train_loss': model.train_loss_history,
        'train_acc': model.train_acc_history,
        'val_loss': model.val_loss_history,
        'val_acc': model.val_acc_history,
    }
    save_history(dataset_name, history)
    save_pickle(f'{dataset_name}_prediction_result.pkl', {
        'pred_y': pred_y,
        'true_y': true_y,
        'scores': score,
        'history': history,
        'config': config,
    })
    print(f'{dataset_name.upper()} test scores: {score}')
    return score


def run_ablation():
    '''Vary depth, hidden width, and dropout, comparing test metrics
    across all three datasets.'''
    print('\n========== ABLATION STUDIES ==========')

    # (label, kwargs overriding the baseline config)
    variants = [
        ('Baseline (2-layer, h=16, drop=0.5)', {}),
        ('1 layer (no hidden)', {'num_layers': 1}),
        ('3 layers', {'num_layers': 3}),
        ('Hidden = 8', {'hidden_dim': 8}),
        ('Hidden = 32', {'hidden_dim': 32}),
        ('Hidden = 64', {'hidden_dim': 64}),
        ('No dropout (drop=0.0)', {'dropout': 0.0}),
    ]

    rows = []
    for dataset_name in ['cora', 'pubmed', 'citeseer']:
        print(f'\n----- ablation on {dataset_name.upper()} -----')
        set_seed()
        data_obj = Dataset_Loader(dName=dataset_name)
        data_obj.dataset_source_folder_path = os.path.join(DATA_DIR, dataset_name)
        data = data_obj.load()
        graph = data['graph']
        features = graph['X']
        adj = graph['utility']['A']
        labels = graph['y']
        splits = data['train_test_val']
        idx_train, idx_val, idx_test = (
            splits['idx_train'], splits['idx_val'], splits['idx_test'])
        true_y = labels[idx_test].cpu().numpy()
        evaluator = Evaluate_Metrics('metrics', '')

        for label, overrides in variants:
            set_seed()
            config = dict(DATASET_CONFIG[dataset_name])
            config.update(overrides)
            model = Method_GCN(
                f'{dataset_name}-gcn-ablation', '',
                num_features=data['num_features'],
                num_classes=data['num_classes'],
                **config,
            )
            model.verbose = False
            pred_y = model.run(features, adj, labels, idx_train, idx_val, idx_test)
            evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
            score = evaluator.evaluate()
            rows.append((dataset_name, label, score))
            print(f'  {label:<36} acc={score["accuracy"]:.4f} '
                  f'prec={score["precision"]:.4f} rec={score["recall"]:.4f} '
                  f'f1={score["f1"]:.4f}')

    csv_path = os.path.join(RESULT_DIR, 'ablation_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('dataset,variant,accuracy,precision,recall,f1\n')
        for dataset_name, label, score in rows:
            f.write(f'{dataset_name},{label},{score["accuracy"]},'
                    f'{score["precision"]},{score["recall"]},{score["f1"]}\n')
    print(f'\nablation results saved to {csv_path}')
    return rows


def write_summary(scores):
    json_path = os.path.join(RESULT_DIR, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)

    txt_path = os.path.join(RESULT_DIR, 'summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Stage 5 GCN Node Classification Summary\n\n')
        f.write('Dataset   | Accuracy | Precision | Recall | F1\n')
        f.write('----------+----------+-----------+--------+-------\n')
        for name, score in scores.items():
            f.write(f'{name:<9} | {score["accuracy"]:.4f}   | '
                    f'{score["precision"]:.4f}    | {score["recall"]:.4f} | '
                    f'{score["f1"]:.4f}\n')
    print(f'\nsummary saved to {txt_path} and {json_path}')


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    scores = {}
    for dataset_name in ['cora', 'pubmed', 'citeseer']:
        scores[dataset_name] = run_dataset(dataset_name)
    write_summary(scores)

    print('\n===== FINAL RESULTS =====')
    print('Dataset   | Accuracy | Precision | Recall | F1')
    for name, score in scores.items():
        print(f'{name:<9} | {score["accuracy"]:.4f}   | '
              f'{score["precision"]:.4f}    | {score["recall"]:.4f} | '
              f'{score["f1"]:.4f}')

    run_ablation()


if __name__ == '__main__':
    main()
