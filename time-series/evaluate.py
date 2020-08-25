import torch
import numpy as np
from sklearn.metrics import roc_auc_score


class Evaluator:
    def __init__(self, model, val_loader, test_loader,
                 log_dir, eval_args={}):
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log_dir = log_dir
        self.eval_args = eval_args

        self.test_auc_log, self.val_auc_log = [], []
        self.best_auc, self.best_val_auc = [-float('inf')] * 2

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            val_auc = self.compute_auc(self.val_loader)
            print('val AUC:', val_auc)
            self.val_auc_log.append((epoch, val_auc))

            test_auc = self.compute_auc(self.test_loader)
            print('test AUC:', test_auc)
            self.test_auc_log.append((epoch, test_auc))
        self.model.train()

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            torch.save({'model': self.model.state_dict()},
                       str(self.log_dir / 'best-model.pth'))

        with (self.log_dir / 'val_auc.txt').open('a') as f:
            print(epoch, val_auc, file=f)
        with (self.log_dir / 'test_auc.txt').open('a') as f:
            print(epoch, test_auc, file=f)

    def compute_auc(self, data_loader):
        y_true, y_score = [], []
        for (val, idx, mask, y, _, cconv_graph) in data_loader:
            score = self.model.predict(
                val, idx, mask, cconv_graph, **self.eval_args)
            y_score.append(score.cpu().numpy())
            y_true.append(y.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        return roc_auc_score(y_true, y_score)
