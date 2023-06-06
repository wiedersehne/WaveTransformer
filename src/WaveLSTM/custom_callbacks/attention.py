# Create custom callbacks for our pytorch-lightning model
#   to view the attention mechanism
import numpy as np
from pytorch_lightning import Callback
import torch
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
        # self.class_labels = class_labels

    def run_callback(self, features, labels, log_name, _trainer, _pl_module):

        # Push features through the model
        _, meta_result = _pl_module(features.clone())

        features = features.view(features.size(0), -1)
        attention = meta_result["attention"]            # [batch_size, attention-hops, num_multiscales]

        # Order samples (rows)
        permute_idx = np.argsort(labels)
        features = features[permute_idx, :].detach().cpu()
        labels = np.asarray(labels[permute_idx].detach().cpu(), dtype=np.int)
        attention = attention[permute_idx, :, :].detach().cpu()

        # Decide how we want to view the hops
        attention = torch.mean(attention, dim=1)     # average over attention hops:   [batch_size, num_multiscales]

        # Nested ordering by biclustering of attention
        attention_new = np.zeros_like(attention)
        features_new = np.zeros_like(features)
        for lbl in np.unique(labels):
            lbl_idx = np.where(labels == lbl)[0]
            atn_cls = attention[lbl_idx, :]
            feat_cls = features[lbl_idx, :]
            biclustering = SpectralBiclustering(n_clusters=(1, 1), method="log", random_state=0)
            biclustering.fit(atn_cls)
            atn_cls = atn_cls[np.argsort(biclustering.row_labels_)]
            feat_cls = feat_cls[np.argsort(biclustering.row_labels_)]
            attention_new[lbl_idx, :]  = atn_cls#[:, np.argsort(biclustering.column_labels_)]
            features_new[lbl_idx, :]  = feat_cls#[:, np.argsort(biclustering.column_labels_)]
        attention = attention_new
        features = features_new

        # Make plot
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 5]})
        # fig.suptitle(f'Attention heatmap {"(hops stacked)" if stack else "(hops averaged)"}')

        # Plot prediction and truth
        vmax = np.min((5, features.max()))
        vmin = np.min((1, features.min()))
        sns.heatmap(features, ax=ax1, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax1.set_title(f'True signal')
        sns.heatmap(attention, ax=ax2, cmap='Blues', vmin=0, yticklabels=False)   # vmax=1
        ax2.set_title(f'Attention (hops averaged)')

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
        ax1.set_ylabel("Samples (ordered by label, then spectral attention clustering)")
        ax1.set_xlabel("Signal index (stacked channels)")
        ax2.set_xlabel(r"$j$")
        plt.tight_layout()

        _trainer.logger.experiment.log({
            log_name: wandb.Image(fig)
        })
        plt.close('all')

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            self.run_callback(features, self.val_labels,
                              "Val:Attention", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            self.run_callback(features, self.test_labels,
                              "Test:Attention", trainer, pl_module)


class MultiResolutionEmbedding(Callback, BaseCallback):
    """
    View the multi-resolution embedding, M
    """
    def __init__(self, val_samples=None, test_samples=None, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module):
        # Push features through the model
        _, meta_result = _pl_module(features)

        wandb_images = []

        # Multi-resolution plot
        M = meta_result["M"]                        # [batch_size, attention-hops, scale_embed_dim]
        Mbar = torch.mean(M, dim=1)                 # average over attention hops:   [batch_size, scale_embed_dim]
        Mbar = np.asarray(Mbar.detach().cpu())

        # Project, if needed
        title = ""
        if (Mbar.shape[1] == 2):
            fig, ax = plt.subplots(1, 1)
        elif (Mbar.shape[1] == 3):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots(1, 1)
            # U-MAP
            Mbar = umap.UMAP(n_components=2, random_state=42).fit_transform(Mbar)
            title = "(U-MAP)"
        fig.suptitle(f"Multi-resolution embedding {title}")
        self.embedding(ax, Mbar, labels=labels)
        plt.tight_layout()
        wandb_images.append(wandb.Image(fig))

        # Multi-resolution w/ 2d U-MAP
        Mbar_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(Mbar)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Multi-resolution embedding (with U-MAP)")
        self.embedding(ax, Mbar_umap, labels=labels)
        plt.tight_layout()
        wandb_images.append(wandb.Image(fig))

        # Combined resolution U-MAP
        # h = meta_result["resolution_embeddings"]
        # h = torch.stack(h, dim=1).view(M.size(0), -1)    #  [batch_size, num_multiscales * n_hidden]
        # latent = umap.UMAP(n_components=2, random_state=42).fit_transform(np.asarray(h.detach().cpu()))
        # fig, ax = plt.subplots(1, 1)
        # fig.suptitle("Stacked resolution embeddings (U-MAP)")
        # self.embedding(ax, latent, labels=labels)
        # plt.tight_layout()
        # wandb_images.append(wandb.Image(fig))

        # Log
        _trainer.logger.experiment.log({
            log_name: wandb_images
        })
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            self.run_callback(features, self.val_labels,
                              "Val:Mbar", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            self.run_callback(features, self.test_labels,
                              "Test:Mbar", trainer, pl_module)