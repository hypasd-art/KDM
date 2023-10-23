"""
The code is copied and adapted from https://arxiv.org/abs/2209.14734
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_ce=torch.tensor(0.)
        self.total_samples=torch.tensor(0.)

    def forward(self, preds: Tensor, target: Tensor):
        """ 
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        res = output/preds.size(0)
        self.total_ce += output.cpu()
        self.total_samples += preds.size(0)
        return res

    def compute(self):
        return self.total_ce / self.total_samples
    
    def reset(self):
        self.total_ce = torch.tensor(0.)
        self.total_samples = torch.tensor(0.)


class TrainLossDiscrete(nn.Module):
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log = False):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)
        # print(mask_E)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = 0.0
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y, loss_X, loss_E

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        print(f"Epoch {current_epoch} finished: X: {epoch_node_loss :.2f} -- E: {epoch_edge_loss :.2f} " f"y: {epoch_y_loss :.2f}")



