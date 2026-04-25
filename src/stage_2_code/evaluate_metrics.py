'''
Concrete Evaluate class - Accuracy, Precision, Recall, F1 (weighted/macro)
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
        accuracy  = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        recall    = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        f1        = f1_score(true_y, pred_y, average='weighted', zero_division=0)

        return {
            'accuracy':  round(accuracy,  4),
            'precision': round(precision, 4),
            'recall':    round(recall,    4),
            'f1':        round(f1,        4)
        }