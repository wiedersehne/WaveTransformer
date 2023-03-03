# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
import wandb
import matplotlib.pyplot as plt
from source.custom_callbacks.base import BaseCallback


class ViewEmbedding(Callback, BaseCallback):
    """
    Callback to view latent embedding  of labelled data at each recurrent step,
     plotting the first two principal components of each latent embedding, and the free-energy of each component
    """
    def __init__(self, val_samples=None, test_samples=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, num_samples=None)

    def run_callback(self, features, labels, sub_labels, log_name, _trainer, _pl_module):
        # Push features through the model
        _, meta_result = _pl_module(features)
        features = np.asarray(features.detach().cpu(), dtype=np.float)

        # Hidden embedding vectors
        wandb_images = []
        for level, z in enumerate(meta_result['hidden']):
            z = z.detach().cpu()

            # Plot first two principal components of the latent space
            (U, S, V) = torch.pca_lowrank(z, niter=2)
            _pc = np.asarray(torch.matmul(z, V[:, :2]))
            _ev = S**2/(z.shape[0]-1)

            fig, (ax, ax_hist) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]})
            fig.suptitle(f"Hidden LSTM embedding, layer {level + 1}")
            self.embedding(ax, _pc, labels=labels, sub_labels=sub_labels,
                           metric=np.mean(features.reshape((features.shape[0], -1)), axis=-1)   # None
                           )
            ax_hist.barh(np.arange(_ev.shape[0]) + 1, _ev, orientation='horizontal')
            ax_hist.set_xlabel("EV")
            plt.tight_layout()

            wandb_images.append(
                wandb.Image(fig)
            )

        _trainer.logger.experiment.log({
            log_name: wandb_images
        })
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            self.run_callback(features, self.val_labels, self.val_sub_labels,
                              "Val:EmbeddingPCA", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            self.run_callback(features, self.test_labels, self.test_sub_labels,
                              "Test:EmbeddingPCA", trainer, pl_module)


class ViewSignal(Callback, BaseCallback):
    """
    Callback on validation epoch end, plotting predictions, as heatmap and as individual line plots
    """
    def __init__(self, val_samples, test_samples, num_samples=8):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module):
        # Push features through the model
        recon, meta_result = _pl_module(features)

        recon = np.asarray(recon.detach().cpu())
        features = np.asarray(features.detach().cpu())
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)

        # Log results
        _trainer.logger.experiment.log({
            log_name:
                [wandb.Image(self.heatmap(recon[:, strand, :, :].reshape(features.shape[0], -1),
                                          features[:, strand, :, :].reshape(features.shape[0], -1),
                                          labels, title=f"Strand index {strand}"))
                 for strand in range(features.shape[1])]
        })
        plt.close('all')

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            if features.dim() == 3:
                features = features.unsqueeze(2)

            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test_Prediction", trainer, pl_module)


class ViewRecurrentSignal(Callback, BaseCallback):
    """
    Callback on validation epoch end, plotting predictions, as heatmap and as individual line plots
    """
    def __init__(self, val_samples, test_samples, num_samples=8):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module):
        # Push features through the model
        _, meta_result = _pl_module(features)
        features = np.asarray(features.detach().cpu())

        recurrent_recon = [np.asarray(x.detach().cpu()) for x in meta_result['pred_recurrent_recon']]
        true_recurrent_recon = [np.asarray(x.detach().cpu()) for x in meta_result['true_recurrent_recon']]
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)

        recurrent_proj_hidden = []
        for z in meta_result['hidden']:
            (U, S, V) = torch.pca_lowrank(z, niter=2)
            recurrent_proj_hidden.append(np.asarray(torch.matmul(z, V).detach().cpu()))

        # Plot only first strand
        # recurrent_recon = [x[:, 0, :, :] for x in recurrent_recon]
        # true_recurrent_recon = [x[:, 0, :, :] for x in true_recurrent_recon]

        _trainer.logger.experiment.log({
            log_name:
                [wandb.Image(self.heatmap(recon_rnn.reshape(features.shape[0], -1),
                                          features.reshape(features.shape[0], -1), #true_rnn.reshape(features.shape[0], -1),
                                          labels,
                                          proj_hidden_rnn,
                                          title=f"Depth index {depth}"))
                 for depth, (recon_rnn, true_rnn, proj_hidden_rnn) in enumerate(zip(recurrent_recon,
                                                                                true_recurrent_recon,
                                                                                recurrent_proj_hidden))]
        })
        plt.close('all')

    def on_test_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.test_features.to(device=pl_module.device)
            if features.dim() == 3:
                features = features.unsqueeze(2)

            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test_RecurrentPrediction", trainer, pl_module)



