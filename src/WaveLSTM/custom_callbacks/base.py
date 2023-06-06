# Base class for callback classes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.cluster.hierarchy as hcluster


class BaseCallback(object):

    def __init__(self, val_samples=None, test_samples=None, label_dict=None):

        assert (val_samples is not None) or (test_samples is not None), "Must supply a validation or test set"

        self.val_features = val_samples['feature'] if val_samples is not None else None
        self.val_labels = val_samples['label'] if val_samples is not None else None

        self.test_features = test_samples['feature'] if test_samples is not None else None
        self.test_labels = test_samples['label'] if test_samples is not None else None

        if self.val_features is not None and self.val_features.dim() != 3:
            size = self.val_features.size()
            self.val_features = self.val_features.view(size[0], -1, size[-1])
        if self.test_features is not None and self.test_features.dim() != 3:
            size = self.test_features.size()
            self.test_features = self.test_features.view(size[0], -1, size[-1])

        self.label_dict = label_dict

    def embedding(self, ax, z, labels):
        """
        Plot a latent embedding on axis `ax`.
            Optionally include labels, or a metric tied to each sample
        """

        col_iterator = iter(get_cmap("tab10").colors)
        for lbl in np.unique(labels):
            mask = np.ma.getmask(np.ma.masked_equal(labels, lbl))
            color = next(col_iterator)
            c = self.label_dict[lbl] if self.label_dict is not None else lbl
            if z.shape[1] == 3:
                ax.scatter(z[mask, 0], z[mask, 1], z[mask,2], c=np.array([color]), label=c, alpha=0.5, edgecolors='none')
            else:
                ax.scatter(z[mask, 0], z[mask, 1], c=np.array([color]), label=c, alpha=0.5, edgecolors='none')


        ax.legend()
        ax.set_xlabel("Embed dim $1$")
        ax.set_ylabel("Embed dim $2$")
        # ax.set_zlabel("Embed dim $3$")

        return ax

    def heatmap(self, prediction, masked_truth, truth, labels, scale=False, title=""):
        """
        Create a subfigure of truth, prediction.
            Options: if pc is None then in 3rd figure we plot the root absolute error, else we plot the pc and order by
                     1st pc
        """
        # Re-order
        permute_idx = np.argsort(labels)
        prediction = prediction[permute_idx, :]
        truth = truth[permute_idx, :]
        masked_truth = masked_truth[permute_idx, :]
        labels = labels[permute_idx]

        # Plot prediction and truth
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        vmax = np.min((5, np.max((truth.max(), prediction.max()))))
        vmin = np.min((0, np.min((truth.min(), prediction.min()))))
        sns.heatmap(prediction, ax=ax1, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax1.set_title(f'Prediction')
        sns.heatmap(masked_truth, ax=ax2, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax2.set_title(f'Masked truth')
        sns.heatmap(truth, ax=ax3, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax3.set_title(f'Truth')

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("Locus")
            ax.set_ylabel(f"Sample (permuted by label)")
            ax.xaxis.set_tick_params(rotation=90)
            for idx, label in enumerate(ax.yaxis.get_ticklabels()):
                if idx % 5 != 0:
                    label.set_visible(False)

        fig.suptitle(title)
        plt.tight_layout()

        return fig
