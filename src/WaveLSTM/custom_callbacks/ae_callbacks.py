# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import wandb
import matplotlib.pyplot as plt
from WaveLSTM.custom_callbacks.base import BaseCallback


class ViewRecurrentSignal(Callback, BaseCallback):
    """
    Callback on validation epoch end, plotting predictions, as heatmap and as individual line plots
    """
    def __init__(self, val_samples, test_samples, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, file_path=None):
        # Push features through the model
        _, meta_result = _pl_module(features)
        features = np.asarray(features.detach().cpu())

        # IWT(0,...,alpha_j, 0....)
        masked_inputs = [np.asarray(x.detach().cpu()) for x in meta_result['masked_inputs']]
        # IWT(alpha_1,...,alpha_j, 0....)
        r_masked_targets = [np.asarray(x.detach().cpu()) for x in meta_result['r_masked_target']]
        r_masked_recons = [np.asarray(x.detach().cpu()) for x in meta_result['r_masked_prediction']]
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)

        recurrent_proj_hidden = []
        for z_idx, z in enumerate(meta_result['hidden']):
            # PCA
            # (U, S, V) = torch.pca_lowrank(z, niter=2)
            # recurrent_proj_hidden.append(np.asarray(torch.matmul(z, V).detach().cpu()))
            # t-SNE
            latent = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3).fit_transform(
                np.asarray(z.detach().cpu()))
            recurrent_proj_hidden.append(latent)


        _trainer.logger.experiment.log({
            log_name + "-right":
                [wandb.Image(self.heatmap(r_masked_recons[j].reshape(features.shape[0], -1),
                                          r_masked_targets[j].reshape(features.shape[0], -1),
                                          labels,
                                          recurrent_proj_hidden[j],
                                          title=f"Right-masked, j={j}"))
                 for j in range(len(r_masked_recons))]
        })
        plt.close('all')

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            if features.dim() == 3:
                features = features.unsqueeze(2)

            labels = self.val_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Validation::RecurrentPrediction", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            if features.dim() == 3:
                features = features.unsqueeze(2)

            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test::RecurrentPrediction", trainer, pl_module)
