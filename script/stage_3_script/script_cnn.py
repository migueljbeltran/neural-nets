'''
Stage 3 CNN experiments on MNIST, ORL, and CIFAR.
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


def save_learning_curve(loss_hist, acc_hist, out_file, title):
    epochs = np.arange(1, len(loss_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, loss_hist, 'b-o', markersize=3)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, np.array(acc_hist) * 100.0, 'g-o', markersize=3)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_one_dataset(dataset_name, dataset_file, use_single_channel, model_kwargs):
    data_obj = Dataset_Loader(dataset_name, '')
    data_obj.dataset_source_folder_path = 'data/stage_3_data/'
    data_obj.dataset_source_file_name = dataset_file
    data_obj.use_single_channel = use_single_channel

    method_obj = Method_CNN(f'cnn-{dataset_name.lower()}', '')
    for k, v in model_kwargs.items():
        setattr(method_obj, k, v)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_3_result/'
    result_obj.result_destination_file_name = f'{dataset_name.lower()}_prediction_result.pkl'

    setting_obj = Setting_Pre_Split('pre-split', '')
    eval_obj = Evaluate_Metrics('metrics', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, eval_obj)

    print(f'\n===== Running {dataset_name} =====')
    scores = setting_obj.load_run_save_evaluate()
    save_learning_curve(
        result_obj.data['train_loss_history'],
        result_obj.data['train_acc_history'],
        f"result/stage_3_result/{dataset_name.lower()}_learning_curve.png",
        f"{dataset_name} CNN Learning Curves",
    )
    print(f'{dataset_name} scores: {scores}')
    return scores


def run_mnist_ablation(name, cfg, base_seed=2):
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = 'data/stage_3_data/'
    data_obj.dataset_source_file_name = 'MNIST'
    data_obj.use_single_channel = True
    loaded = data_obj.load()

    # Use subset for faster ablation while preserving ranking trends.
    X_train = loaded['train']['X'][:12000]
    y_train = loaded['train']['y'][:12000]
    X_test = loaded['test']['X'][:3000]
    y_test = loaded['test']['y'][:3000]

    method_obj = Method_CNN(f'mnist-{name}', '')
    base_cfg = {
        'conv_channels': [32, 64],
        'hidden_dim': 128,
        'max_epoch': 4,
        'batch_size': 128,
        'optimizer_name': 'adam',
        'loss_name': 'cross_entropy',
        'learning_rate': 0.001,
    }
    base_cfg.update(cfg)
    for k, v in base_cfg.items():
        setattr(method_obj, k, v)

    pred = method_obj.run(X_train, y_train, X_test)
    eval_obj = Evaluate_Metrics('metrics', '')
    eval_obj.data = {'pred_y': pred, 'true_y': y_test}
    scores = eval_obj.evaluate()
    save_learning_curve(
        method_obj.train_loss_history,
        method_obj.train_acc_history,
        f'result/stage_3_result/mnist_{name}_learning_curve.png',
        f'MNIST Ablation - {name}',
    )
    print(f'MNIST_{name} scores: {scores}')
    return scores


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    os.makedirs('result/stage_3_result/', exist_ok=True)

    all_scores = {}

    all_scores['MNIST'] = run_one_dataset(
        dataset_name='MNIST',
        dataset_file='MNIST',
        use_single_channel=True,
        model_kwargs={
            'conv_channels': [32, 64],
            'hidden_dim': 128,
            'max_epoch': 8,
            'batch_size': 128,
            'optimizer_name': 'adam',
            'loss_name': 'cross_entropy',
            'learning_rate': 0.001,
        },
    )

    all_scores['ORL'] = run_one_dataset(
        dataset_name='ORL',
        dataset_file='ORL',
        use_single_channel=True,
        model_kwargs={
            'conv_channels': [16, 32],
            'hidden_dim': 128,
            'max_epoch': 20,
            'batch_size': 32,
            'optimizer_name': 'adam',
            'loss_name': 'cross_entropy',
            'learning_rate': 0.001,
        },
    )

    all_scores['CIFAR'] = run_one_dataset(
        dataset_name='CIFAR',
        dataset_file='CIFAR',
        use_single_channel=False,
        model_kwargs={
            'conv_channels': [32, 64, 128],
            'hidden_dim': 256,
            'max_epoch': 10,
            'batch_size': 128,
            'optimizer_name': 'adam',
            'loss_name': 'cross_entropy',
            'learning_rate': 0.001,
        },
    )

    # 3-5: configuration impact study on MNIST
    print('\n===== MNIST configuration impact study =====')
    ablation_configs = [
        ('deeper-net', {'conv_channels': [32, 64, 128], 'hidden_dim': 256, 'max_epoch': 8}),
        ('larger-kernel', {'kernel_size': 5, 'padding': 2, 'conv_channels': [32, 64], 'max_epoch': 4}),
        ('sgd-optimizer', {'optimizer_name': 'sgd', 'learning_rate': 0.01, 'max_epoch': 4}),
        ('mse-loss', {'loss_name': 'mse', 'conv_channels': [32, 64], 'max_epoch': 4}),
    ]

    for name, cfg in ablation_configs:
        scores = run_mnist_ablation(name, cfg)
        all_scores[f'MNIST_{name}'] = scores

    print('\n===== Final Summary =====')
    for k, v in all_scores.items():
        print(f'{k}: acc={v["accuracy"]:.4f}, f1_weighted={v["f1_weighted"]:.4f}, f1_macro={v["f1_macro"]:.4f}')
