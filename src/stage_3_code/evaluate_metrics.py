'''
Evaluation metrics for multiclass stage 3 classification.
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription

    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        return {
            'accuracy': round(accuracy_score(true_y, pred_y), 4),
            'precision_macro': round(precision_score(true_y, pred_y, average='macro', zero_division=0), 4),
            'recall_macro': round(recall_score(true_y, pred_y, average='macro', zero_division=0), 4),
            'f1_macro': round(f1_score(true_y, pred_y, average='macro', zero_division=0), 4),
            'precision_weighted': round(precision_score(true_y, pred_y, average='weighted', zero_division=0), 4),
            'recall_weighted': round(recall_score(true_y, pred_y, average='weighted', zero_division=0), 4),
            'f1_weighted': round(f1_score(true_y, pred_y, average='weighted', zero_division=0), 4),
        }
