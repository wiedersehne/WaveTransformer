# Create custom callbacks for our pytorch-lightning model
#   to view the attention mechanism
import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.cluster import SpectralBiclustering

from WaveLSTM.custom_callbacks.base import BaseCallback


class Attention(Callback, BaseCallback):
    """
    Similar to viewing the sentence embedding (see "a structured self-attentive sentence embedding", Bengio)
    """
    def __init__(self, val_samples=None, test_samples=None, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, order_method="max", **kwargs):

        # Push features through the model
        _, meta_result = _pl_module(features.clone(), **kwargs)

        features = features.view(features.size(0), -1)
        attention = meta_result["attention"]            # [batch_size, attention-hops, num_multiscales]

        # Order samples (rows)
        permute_idx = np.argsort(labels)
        features = features[permute_idx, :].detach().cpu()
        labels = np.asarray(labels[permute_idx].detach().cpu(), dtype=np.int)
        attention = attention[permute_idx, :, :].detach().cpu()

        # Decide how we want to view over the hops
        attention = torch.mean(attention, dim=1)     # average over attention hops:   [batch_size, num_multiscales]

        # Nested ordering by <order_method> on attention matrix
        attention_new = np.zeros_like(attention)
        features_new = np.zeros_like(features)
        for lbl in np.unique(labels):
            lbl_idx = np.where(labels == lbl)[0]
            atn_cls = attention[lbl_idx, :]
            feat_cls = features[lbl_idx, :]
            if order_method == "bispectral":
                # Make and fit spectral bi-clustering algorithm with pre-defined number of clusters in each direction
                biclustering = SpectralBiclustering(n_clusters=(2, 1), method="log", random_state=0)
                biclustering.fit(atn_cls)
                # Sort attention and feature rows
                atn_cls = atn_cls[np.argsort(biclustering.row_labels_)]
                feat_cls = feat_cls[np.argsort(biclustering.row_labels_)]
                attention_new[lbl_idx, :]  = atn_cls#[:, np.argsort(biclustering.column_labels_)]
                features_new[lbl_idx, :]  = feat_cls#[:, np.argsort(biclustering.column_labels_)]
            elif order_method == "max":
                idx = np.argsort(np.argmax(atn_cls, axis=1))
                attention_new[lbl_idx, :] = atn_cls[idx]
                features_new[lbl_idx, :] = feat_cls[idx, :]
            else:
                raise NotImplementedError
        attention = attention_new
        features = features_new

        # Make plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [8, 5]})
        # fig.suptitle(f'Attention heatmap {"(hops stacked)" if stack else "(hops averaged)"}')

        # Plot prediction and truth
        vmax = np.min((5, features.max()))
        vmin = np.min((1, features.min()))
        sns.heatmap(features, ax=ax1, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax1.set_title(f'True signal')
        sns.heatmap(attention, ax=ax2, cmap='Blues', vmin=0, yticklabels=False)   # vmax=1
        ax2.set_title(r'Attention ($\bar{m}$)')

        # Format
        y_tick_locations = [np.where(labels == lbl)[0][0] for lbl in np.unique(labels)]
        for ax in [ax1, ax2]:
            ax.set_yticks(y_tick_locations)
            ax.set_yticks([], minor=True)
            ax.yaxis.grid(True, which='major', lw=1, ls='-', color='k')
        if self.label_dict is None:
            ax1.set_yticklabels(np.unique(labels))
        else:
            ax1.set_yticklabels([self.label_dict[i] for i in np.unique(labels)])
        ax1.set_xticklabels([], minor=False)

        ax1.set_xticks([])
        ax1.set_ylabel(f"Samples (permuted)")
        ax1.set_xlabel("Loci (stacked)")

        ax2.set_xlabel(r"$j$")

        plt.tight_layout()

        _trainer.logger.experiment.log({
            log_name: wandb.Image(fig)
        })
        plt.close('all')

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            # Send to device
            features = self.val_features.to(device=pl_module.device)
            val_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in self.val_surv.items()}    # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.val_labels, "Val:Attention", trainer, pl_module, **val_surv)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in self.test_surv.items()}    # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, "Test:Attention", trainer, pl_module, **test_surv)


class MultiResolutionEmbedding(Callback, BaseCallback):
    """
    View the multi-resolution embedding, M
    """
    def __init__(self, val_samples=None, test_samples=None, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, proj="tsne", proj_3d=True, **kwargs):
        # Push features through the model
        _, meta_result = _pl_module(features, **kwargs)

        wandb_images = []

        # Multi-resolution plot
        M = meta_result["M"]                        # [batch_size, attention-hops, scale_embed_dim]
        Mbar = torch.mean(M, dim=1)                 # average over attention hops:   [batch_size, scale_embed_dim]
        Mbar = np.asarray(Mbar.detach().cpu())

        # Plot depends on shape of latent dimension
        if Mbar.shape[1] == 1:
            # Histogram for 1-d embedding
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Multi-resolution KDE")
            self.histogram(ax, Mbar[:, 0], labels=labels, xlabel=r"Latent multi-resolution embedding $(\bar{m})$")
        elif (Mbar.shape[1] == 2):
            # 2d scatter plot for 2-d embedding
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f"Multi-resolution embedding")
            self.embedding(ax, Mbar, labels=labels)
        elif (Mbar.shape[1] == 3) and (proj_3d == False):
            # 3d scatter plot for 3-d embedding
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            fig.suptitle(f"Multi-resolution embedding")
            self.embedding(ax, Mbar, labels=labels)
        else:
            # Else we project with U-MAP to 2-d and do a 2d scatter
            fig, ax = plt.subplots(1, 1)
            if proj == "umap":
                fig.suptitle(f"Multi-resolution embedding (U-MAP)")
                Mbar_proj = umap.UMAP(n_components=2).fit_transform(Mbar)  # random_state=42
            else:
                perp = np.max((3, np.min((30, int(0.1 * Mbar.shape[0])))))
                fig.suptitle(f"Multi-resolution embedding (t-SNE, perplexity={perp})")
                Mbar_proj = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(Mbar)
            self.embedding(ax, Mbar_proj, labels=labels)

        plt.tight_layout()
        wandb_images.append(wandb.Image(fig))

        # Log
        _trainer.logger.experiment.log({
            log_name: wandb_images
        })
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            # Send to device
            features = self.val_features.to(device=pl_module.device)
            val_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.val_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.val_labels, "Val:MultiResolution", trainer, pl_module, **val_surv)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, "Test:MultiResolution", trainer, pl_module, **test_surv)
