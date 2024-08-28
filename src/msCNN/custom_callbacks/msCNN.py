# Create custom callbacks for our pytorch-lightning model

import numpy as np
import pickle

import sklearn.manifold
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from src.msCNN.custom_callbacks.base import BaseCallback


def on_validation_epoch_end(self, trainer, pl_module):
    if self.val_features is not None:
            # Send to device
            features = self.val_features.to(device=pl_module.device)
            val_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.val_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.val_labels, "Val:Resolution", trainer, pl_module, **val_surv)

def on_test_epoch_end(self, trainer, pl_module):
    if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, "Test:Resolution", trainer, pl_module, **test_surv)

class SaveOutput(Callback, BaseCallback):
    """
    Callback on test epoch end to save outputs for plotting
    """
    def __init__(self, test_samples, file_path=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, test_samples=test_samples)
        self.file_path = file_path if file_path is not None else "output.pkl"

    def run_callback(self, features, labels, _pl_module, **kwargs):
        # Push features through the model
        recon, meta_result = _pl_module(features, **kwargs)
        meta_result["labels"] = labels

        with open(self.file_path, 'wb') as file:
            pickle.dump(meta_result, file)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, pl_module, **test_surv)