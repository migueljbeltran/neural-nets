'''
Evaluation metrics for Stage 4 binary text classification.
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        return {
            'accuracy': round(accuracy_score(true_y, pred_y), 4),
            'precision': round(precision_score(true_y, pred_y, zero_division=0), 4),
            'recall': round(recall_score(true_y, pred_y, zero_division=0), 4),
            'f1': round(f1_score(true_y, pred_y, zero_division=0), 4),
        }
