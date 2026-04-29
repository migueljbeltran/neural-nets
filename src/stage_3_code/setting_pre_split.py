'''
Pre-split setting for stage 3 datasets.
'''

from src.base_class.setting import setting


class Setting_Pre_Split(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train = data['train']['X'], data['train']['y']
        X_test, y_test = data['test']['X'], data['test']['y']

        pred_y = self.method.run(X_train, y_train, X_test)
        self.result.data = {
            'pred_y': pred_y,
            'true_y': y_test,
            'train_loss_history': self.method.train_loss_history,
            'train_acc_history': self.method.train_acc_history,
        }
        self.result.save()

        self.evaluate.data = {'pred_y': pred_y, 'true_y': y_test}
        return self.evaluate.evaluate()
