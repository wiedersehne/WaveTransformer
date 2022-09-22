import numpy as np
from pytorch_lightning import Callback
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


# Create custom callback for the end of each validation epoch
class LatentSpace(Callback):
    """
    Callback on validation epoch end, plotting the first two principal components of the variational latent means
    """
    def __init__(self, val_samples):
        super().__init__()
        self.val_features = val_samples['feature']
        self.val_labels = val_samples['label']

    @staticmethod
    def embedding(x, labels, annotated=True):
        # standardise
        x -= x.min(axis=0)
        x /= x.max(axis=0)
        fig, ax = plt.subplots()
        if annotated:
            for n in range(x.shape[0]):
                ax.annotate(str(labels[n]), (x[n, 0], x[n, 1]))
        else:
            ax.scatter(list(x[:, 0]), list(x[:, 1]))
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        plt.tight_layout()
        return fig

    def on_validation_epoch_end(self, trainer, pl_module):
        val_features = self.val_features.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Push validation features through model
        recon_val, x_val, mu_val, log_var_val = pl_module(val_features)

        # Plot first two principal components of the latent space
        (U, S, V) = torch.pca_lowrank(mu_val, niter=2)
        pc = np.asarray(torch.matmul(mu_val, V[:, :2]).detach().cpu())
        # eigenvalues = S**2 / (U.size(0) - 1)
        labels = np.asarray(val_labels.detach().cpu(), dtype=np.int)
        trainer.logger.experiment.log({
            "Validation_LatentSpacePCA":  wandb.Image(self.embedding(pc, labels))
        })


class FeatureSpace1d(Callback):
    """
    Callback on validation epoch end, plotting the first two principal components of the variational latent means
    """
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_features = val_samples['feature']
        self.val_labels = val_samples['label']
        assert num_samples < self.val_features.size(0)

    @staticmethod
    def heatmap(prediction, truth, labels):
        permute_idx = np.argsort(labels)
        prediction = prediction[permute_idx, :]
        truth = truth[permute_idx, :]
        labels = labels[permute_idx]

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
        vmax = np.max((truth.max(), prediction.max()))
        sns.heatmap(prediction, ax=ax1, cmap='Blues', vmax=vmax, yticklabels=labels)
        sns.heatmap(truth, ax=ax2, cmap='Blues', vmax=vmax, yticklabels=labels)
        for ax in [ax1, ax2]:
            ax.set_xlabel("Locus")
            ax.set_ylabel("Permuted index (by label)")
            ax.xaxis.set_tick_params(rotation=90)
            for idx, label in enumerate(ax.yaxis.get_ticklabels()):
                if idx % 5 != 0:
                    label.set_visible(False)

        ax1.set_title('Prediction')
        ax2.set_title('Truth')
        plt.tight_layout()
        return fig

    def on_validation_epoch_end(self, trainer, pl_module):
        val_features = self.val_features.to(device=pl_module.device)    # TODO: Needed?
        val_labels = self.val_labels.to(device=pl_module.device)
        # Push validation features through model
        recon_val, x_val, mu_val, log_var_val = pl_module(val_features)

        recon_val = np.asarray(recon_val.detach().cpu())
        x_val = np.asarray(x_val.detach().cpu())
        val_labels_det = np.asarray(val_labels.detach().cpu(), dtype=np.int)

        trainer.logger.experiment.log({
            "Validation_FeatureSpace1d": wandb.Image(self.heatmap(recon_val, x_val, val_labels_det))
        })
