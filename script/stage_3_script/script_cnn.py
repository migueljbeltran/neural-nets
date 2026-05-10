'''
Stage 3 CNN experiments on MNIST, ORL, and CIFAR.

Run from the repo root:
    python -m script.stage_3_script.script_cnn
'''

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.stage_3_code.dataset_loader import Dataset_Loader
from src.stage_3_code.method_cnn import Method_CNN
from src.stage_3_code.result_saver import Result_Saver
from src.stage_3_code.setting_pre_split import Setting_Pre_Split
from src.stage_3_code.evaluate_metrics import Evaluate_Metrics


SEED = 2
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'stage_3_data') + os.sep
RESULT_DIR = os.path.join(REPO_ROOT, 'result', 'stage_3_result') + os.sep


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_learning_curve(loss_hist, acc_hist, out_file, title):
    epochs = np.arange(1, len(loss_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, loss_hist, 'b-o', markersize=3)
    ax1.set(title='Training Loss', xlabel='Epoch', ylabel='Loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, np.array(acc_hist) * 100.0, 'g-o', markersize=3)
    ax2.set(title='Training Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_experiment(name, dataset_file, *,
                   single_channel,
                   model_kwargs,
                   normalize=False,
                   title=None):
    set_seed()
    title = title or f'{name} CNN Learning Curves'

    data_obj = Dataset_Loader(name, '',
                              use_single_channel=single_channel,
                              normalize=normalize)
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_source_file_name = dataset_file

    method_obj = Method_CNN(f'cnn-{name.lower()}', '', **model_kwargs)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_DIR
    result_obj.result_destination_file_name = f'{name.lower()}_prediction_result.pkl'

    setting_obj = Setting_Pre_Split('pre-split', '')
    eval_obj = Evaluate_Metrics('metrics', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, eval_obj)

    print(f'\n===== {name} =====')
    scores = setting_obj.load_run_save_evaluate()
    save_learning_curve(
        result_obj.data['train_loss_history'],
        result_obj.data['train_acc_history'],
        os.path.join(RESULT_DIR, f'{name.lower()}_learning_curve.png'),
        title,
    )
    print(f'{name} scores: {scores}')
    return scores


# Main per-dataset configurations (3-3, 3-4).
MAIN_RUNS = [
    dict(name='MNIST', dataset_file='MNIST', single_channel=True, model_kwargs=dict(
        conv_channels=[32, 64], hidden_dim=128, max_epoch=8, batch_size=128,
        optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    )),
    dict(name='ORL', dataset_file='ORL', single_channel=True, model_kwargs=dict(
        conv_channels=[16, 32], hidden_dim=128, max_epoch=20, batch_size=32,
        optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    )),
    dict(name='CIFAR', dataset_file='CIFAR', single_channel=False, model_kwargs=dict(
        conv_channels=[32, 64, 128], hidden_dim=256, max_epoch=10, batch_size=128,
        optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    )),
]

# MNIST configuration-impact study (3-5).
# All ablations share the same baseline so single-knob effects are isolated.
# 4 epochs keeps the sweep tractable on CPU; "baseline" entry uses the same
# 4-epoch budget so it is directly comparable to the rest of the table.
ABLATION_BASE = dict(
    conv_channels=[32, 64], hidden_dim=128, max_epoch=4, batch_size=128,
    optimizer_name='adam', loss_name='cross_entropy', learning_rate=0.001,
    kernel_size=3, padding=1, stride=1, pool='max', pool_kernel=2,
)

ABLATIONS = [
    ('baseline',      {}),
    ('deeper-net',    dict(conv_channels=[32, 64, 128])),
    ('wider-hidden',  dict(hidden_dim=256)),
    ('larger-kernel', dict(kernel_size=5, padding=2)),
    ('no-padding',    dict(padding=0)),
    ('larger-stride', dict(stride=2)),
    ('avg-pool',      dict(pool='avg')),
    ('mse-loss',      dict(loss_name='mse')),
    ('sgd-optimizer', dict(optimizer_name='sgd', learning_rate=0.01)),
]


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_scores = {}

    for run in MAIN_RUNS:
        all_scores[run['name']] = run_experiment(**run)

    print('\n===== MNIST configuration impact study =====')
    for name, override in ABLATIONS:
        cfg = {**ABLATION_BASE, **override}
        scores = run_experiment(
            name=f'MNIST_{name}',
            dataset_file='MNIST',
            single_channel=True,
            model_kwargs=cfg,
            title=f'MNIST Ablation - {name}',
        )
        all_scores[f'MNIST_{name}'] = scores

    print('\n===== Final Summary =====')
    for k, v in all_scores.items():
        print(f'{k}: acc={v["accuracy"]:.4f}, '
              f'f1_weighted={v["f1_weighted"]:.4f}, '
              f'f1_macro={v["f1_macro"]:.4f}')


if __name__ == '__main__':
    main()
