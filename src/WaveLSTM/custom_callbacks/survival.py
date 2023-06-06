# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from WaveLSTM.custom_callbacks.base import BaseCallback

class PerformanceMetrics(Callback, BaseCallback):
    """
    Record metrics for survival model.
    """
    def get_ctd(self):
        # Get time dependent Concordance Index
        return torch.ones(1).to(torch.float32)

    def get_ibs(self):
        # Get Integrated Brier Score
        return torch.ones(1).to(torch.float32)

    def get_inbll(self):
        # Get Integrated Negative Binomial LogLikelihood
        return torch.ones(1).to(torch.float32)

    def get_mae_rmse(self):
        # Get Mean Absolute Error and Root Mean Square Error
        return torch.ones(1).to(torch.float32), torch.ones(1).to(torch.float32)

    def __init__(self, val_samples, test_samples, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

        for idx, batch in enumerate([val_samples, test_samples]):
            if batch is not None:
                c = torch.stack((batch["days_since_birth"],
                                 torch.tensor([1 if i == "male" else 0 for i in batch['sex']])), dim=1)
                t = batch['survival_time']
                k = batch['survival_status']

                if idx == 0:
                    self.val_ctk = (c, t, k)
                else:
                    self.test_ctk = (c, t, k)

    def run_callback(self, features, c, t, k, labels, log_name, _trainer, _pl_module, file_path=None):
        # Push features through the model
        # _, meta_result = _pl_module(features, c, t, k)

        # _pl_module.predict_step(features, c, t, k)

        ctd = self.get_ctd()
        ibs = self.get_ibs()
        inbll = self.get_inbll()
        mae, rmse = self.get_mae_rmse()

        # Log all
        self.log_dict({"ctd": ctd, "ibs": ibs, "inbll": inbll, "mae": mae, "rmse": rmse})

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            c, t, k = self.val_ctk[0], self.val_ctk[1], self.val_ctk[2]
            labels = self.val_labels.to(device=pl_module.device)
            self.run_callback(features, c, t, k, labels, "Validation::SurvivalMetrics", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            c, t, k = self.test_ctk[0], self.test_ctk[1], self.test_ctk[2]
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, c, t, k, labels, "Test::SurvivalMetrics", trainer, pl_module)
