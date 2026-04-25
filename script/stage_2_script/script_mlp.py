'''
Stage 2 - MLP on MNIST digit classification
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

from src.stage_2_code.dataset_loader    import Dataset_Loader
from src.stage_2_code.method_mlp        import Method_MLP
from src.stage_2_code.result_saver      import Result_Saver
from src.stage_2_code.Setting_Pre_Split import Setting_Pre_Split
from src.stage_2_code.evaluate_metrics  import Evaluate_Metrics

np.random.seed(2)
torch.manual_seed(2)

# ---- object initialization -----------------------------------------------
data_obj = Dataset_Loader('MNIST', '')
data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
data_obj.dataset_train_file_name    = 'train.csv'
data_obj.dataset_test_file_name     = 'test.csv'

method_obj = Method_MLP('MLP', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_2_result/'
result_obj.result_destination_file_name   = 'MLP_prediction_result'

setting_obj  = Setting_Pre_Split('pre-split', '')
evaluate_obj = Evaluate_Metrics('metrics', '')

# ---- run -----------------------------------------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
scores = setting_obj.load_run_save_evaluate()

print('************ Overall Performance ************')
for k, v in scores.items():
    print(f'  {k.capitalize():<12}: {v:.4f}')
print('************ Finish ************')

# ---- convergence plot ----------------------------------------------------
os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)
epochs = range(1, method_obj.max_epoch + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, method_obj.train_loss_history, 'b-o', markersize=4)
ax1.set_title('Training Loss Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, [a * 100 for a in method_obj.train_acc_history], 'g-o', markersize=4)
ax2.set_title('Training Accuracy Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_ylim([0, 101])
ax2.grid(True, alpha=0.3)

plt.suptitle('Stage 2 MLP - MNIST Digit Classification', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(result_obj.result_destination_folder_path + 'learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- ablation study ------------------------------------------------------
data    = data_obj.data
X_train = data['train']['X']
y_train = data['train']['y']
X_test  = data['test']['X']
y_test  = data['test']['y']

print('\n************ Ablation Study ************')
print(f"  {'Config':<38} {'Acc':>6}  {'F1':>6}")
print('  ' + '-' * 52)

def ablation_run(name, model):
    torch.manual_seed(2)
    pred = model.run(X_train, y_train, X_test)
    evaluate_obj.data = {'true_y': y_test, 'pred_y': pred}
    s = evaluate_obj.evaluate()
    print(f"  {name:<38} {s['accuracy']:>6.4f}  {s['f1']:>6.4f}")

# Architecture
ablation_run('Baseline (512-256-128, dropout=0.3)', Method_MLP('', ''))

m = Method_MLP('', '')
m.model = nn.Sequential(
    nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
    nn.Linear(128, 10)
)
ablation_run('Shallow (256-128, dropout=0.3)', m)

m = Method_MLP('', '')
m.model = nn.Sequential(
    nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
    nn.Linear(128, 10)
)
ablation_run('No dropout (512-256-128)', m)

# Optimizer
m = Method_MLP('', '')
m.optimizer_class  = torch.optim.SGD
m.optimizer_kwargs = {'momentum': 0.9}
ablation_run('SGD + momentum=0.9', m)

# Loss function — MSELoss requires one-hot targets instead of class indices
class MSEMethod(Method_MLP):
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
                y_onehot = nn.functional.one_hot(y_batch, num_classes=10).float()
                loss = self.loss_function(torch.softmax(outputs, dim=1), y_onehot)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
                correct    += (outputs.argmax(dim=1) == y_batch).sum().item()
                total      += X_batch.size(0)
            self.train_loss_history.append(epoch_loss / total)
            self.train_acc_history.append(correct / total)

m = MSEMethod('', '')
m.loss_function = nn.MSELoss()
ablation_run('MSELoss', m)

print('****************************************\n')
