# Create custom callbacks for our pytorch-lightning model
#   to view the attention mechanism
import numpy as np
from pytorch_lightning import Callback
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from source.custom_callbacks.base import BaseCallback


class MultiScaleEmbedding(Callback, BaseCallback):
    """
    Similar to viewing the sentence embedding (see "a structured self-attentive sentence embedding", Bengio), here we
    view the multiscale sequence embedding
    """
    def __init__(self, val_samples=None, test_samples=None, class_labels=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, num_samples=None)
        self.class_labels = class_labels

    def run_callback(self, features, labels, sub_labels, log_name, _trainer, _pl_module, sort="label", stack=False):
        # Push features through the model
        _, meta_result = _pl_module(features)

        features = features.view(features.size(0), -1)
        attention = meta_result["attention"]            # [batch_size, attention-hops, num_multiscales]

        # Order samples (rows)
        if sort == "label":
            permute_idx = np.argsort(labels)
        else:
            raise NotImplementedError
        features = features[permute_idx, :].detach().cpu()
        labels = np.asarray(labels[permute_idx].detach().cpu(), dtype=np.int)
        attention = attention[permute_idx, :, :].detach().cpu()

        # Decide how we want to view the hops
        if stack is False:
            attention = torch.mean(attention, dim=1)     # average over attention hops:   [batch_size, num_multiscales]
        else:
            attention = attention.reshape((attention.shape[0]*attention.shape[1], -1))   # stack hops

        # Make plot
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 5]})
        # fig.suptitle(f'Attention heatmap {"(hops stacked)" if stack else "(hops averaged)"}')

        # Plot prediction and truth
        vmax = np.min((5, features.max()))
        vmin = np.min((1, features.min()))
        sns.heatmap(features, ax=ax1, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax1.set_title(f'True signal')
        sns.heatmap(attention, ax=ax2, cmap='Blues', vmin=0, yticklabels=False)   # vmax=1
        ax2.set_title(f'Attention {"(hops stacked)" if stack else "(hops averaged)"}')

        if sort == "label":
            y_tick_locations = [np.where(labels == lbl)[0][0] for lbl in np.unique(labels)]
            for ax in [ax1, ax2]:
                ax.set_yticks(y_tick_locations)
                ax.set_yticks([], minor=True)
                ax.yaxis.grid(True, which='major', lw=1, ls='-', color='k')
            ax1.set_yticklabels(np.unique(labels) if self.class_labels is None else self.class_labels, minor=False)
            ax1.set_xticklabels([], minor=False)

        ax1.set_xticks([])
        ax1.set_ylabel(f"Samples (ordered by {sort})")
        ax1.set_xlabel(r"Signal index (stacked channels)")
        ax2.set_xlabel(r"$\hat{j}$")
        plt.tight_layout()

        _trainer.logger.experiment.log({
            log_name: wandb.Image(fig)
        })
        plt.close('all')

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            self.run_callback(features, self.val_labels, self.val_sub_labels,
                              "Val:Attention", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            self.run_callback(features, self.test_labels, self.test_sub_labels,
                              "Test:Attention", trainer, pl_module)

