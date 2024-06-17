# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from WaveLSTM.custom_callbacks.base import BaseCallback


class RecurrentReconstruction(Callback, BaseCallback):
    """
    For non-attentive autoencoder.
    Plott prediction at each stage
    """
    def __init__(self, val_samples, test_samples, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, file_path=None):
        # Push features through the model
        _, meta_result = _pl_module(features.clone())

        features = np.asarray(features.detach().cpu())
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)
        masked_targets = meta_result['masked_targets']
        masked_recons = meta_result['masked_predictions']

        _trainer.logger.experiment.log({
            log_name + "-right":
                [wandb.Image(self.heatmap(masked_recons[j].reshape(features.shape[0], -1),
                                          masked_targets[j].reshape(features.shape[0], -1),
                                          features.reshape(features.shape[0], -1),
                                          labels,
                                          title=f"Recursive reconstruction, j={j} (channels stacked)"))
                 for j in range(len(masked_recons))]
        })
        plt.close('all')

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            labels = self.val_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Validation::RecurrentReconstruction", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test::RecurrentReconstruction", trainer, pl_module)



class Reconstruction(Callback, BaseCallback):
    """
    Callback on validation epoch end, plotting predictions, as heatmap and as individual line plots
    """
    def __init__(self, val_samples, test_samples, num_samples=8):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module):


        # Push features through the model
        recon, meta_result = _pl_module(features.clone())

        features = np.asarray(features.detach().cpu())
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)
        target = meta_result['masked_targets'][-1]
        recon = np.asarray(recon.detach().cpu())

        # Log results
        _trainer.logger.experiment.log({
            log_name:
                [wandb.Image(self.heatmap(recon[:, channel, :],
                                          target[:, channel, :],
                                          features[:, channel, :],
                                          labels, title=f"Masked-target. Channel: {channel}"))
                 for channel in range(target.shape[1])]
        })
        plt.close('all')

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            labels = self.val_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Validation::Reconstruction", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test::Reconstruction", trainer, pl_module)

# class Reconstruction(Callback, BaseCallback):
#     """
#     Plot prediction
#     """
#     def __init__(self, val_samples, test_samples, label_dictionary=None):
#         Callback.__init__(self)
#         BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)
#
#     def run_callback(self, features, labels, log_name, _trainer, _pl_module, file_path=None):
#         # Push features through the model
#         _, meta_result = _pl_module(features)
#         features = np.asarray(features.detach().cpu())
#         r_masked_target = meta_result['r_masked_target'].detach().cpu()
#
#         M = meta_result["M"]  # [batch_size, attention-hops, scale_embed_dim]
#         Mbar = torch.mean(M, dim=1)  # average over attention hops:   [batch_size, scale_embed_dim]
#         latent = umap.UMAP(n_components=2, random_state=42).fit_transform(np.asarray(M_bottle.detach().cpu()))
#
#         # IWT(0,...,alpha_j, 0....)
#         masked_inputs = [np.asarray(x.detach().cpu()) for x in meta_result['masked_inputs']]
#         # IWT(alpha_1,...,alpha_j, 0....)
#         r_masked_recons = [np.asarray(x.detach().cpu()) for x in meta_result['r_masked_predictions']]
#         labels = np.asarray(labels.detach().cpu(), dtype=np.int)
#
#         fig = self.heatmap(r_masked_recons[j].reshape(features.shape[0], -1),
#                                           r_masked_targets.reshape(features.shape[0], -1),
#                                           labels,
#                                           recurrent_proj_hidden[j],
#                                           title=f"xx")
#
#         _trainer.logger.experiment.log({
#             log_name: wandb.Image(fig)
#         })
#         plt.close('all')
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         if self.val_features is not None:
#             features = self.val_features.to(device=pl_module.device)
#             if features.dim() == 3:
#                 features = features.unsqueeze(2)
#
#             labels = self.val_labels.to(device=pl_module.device)
#             self.run_callback(features, labels, "Validation::Reconstruction", trainer, pl_module)
#
#     def on_test_epoch_end(self, trainer, pl_module):
#         if self.test_features is not None:
#             features = self.test_features.to(device=pl_module.device)
#             if features.dim() == 3:
#                 features = features.unsqueeze(2)
#
#             labels = self.test_labels.to(device=pl_module.device)
#             self.run_callback(features, labels, "Test::Reconstruction", trainer, pl_module)
