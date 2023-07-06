# Base class for callback classes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import scipy.cluster.hierarchy as hcluster
import torch


class BaseCallback(object):

    def __init__(self, val_samples=None, test_samples=None, label_dict=None):

        assert (val_samples is not None) or (test_samples is not None), "Must supply a validation or test set"
        self.label_dict = label_dict

        # Unpack validation hook set
        if val_samples is not None:
            self.val_features = val_samples['feature']         # If we pass in val hook set, there must be features
            assert self.val_features.dim() == 3
            # Optional
            self.val_labels = val_samples['label'] if "label" in val_samples.keys() else None
            # Survival
            c = torch.stack((val_samples["days_since_birth"],
                             torch.tensor([1 if i == "male" else 0 for i in val_samples['sex']])), dim=1) \
                if ("days_since_birth" in val_samples.keys()) and ("sex" in val_samples.keys()) else None
            t =  val_samples['survival_time'] if "survival_time" in val_samples.keys() else None
            k =  val_samples['survival_status'] if "survival_status" in val_samples.keys() else None
            val_surv = {"c": c, "t": t, "k": k}
            self.val_surv = val_surv if None not in val_surv.values() else {}

        # Unpack test hook set
        if test_samples is not None:
            self.test_features = test_samples['feature']       # If we pass in val hook set, there must be features
            assert self.test_features.dim() == 3
            # Optional
            self.test_labels = test_samples['label'] if "label" in test_samples.keys() else None
            # Survival
            c = torch.stack((test_samples["days_since_birth"],
                             torch.tensor([1 if i == "male" else 0 for i in test_samples['sex']])), dim=1) \
                if ("days_since_birth" in test_samples.keys()) and ("sex" in test_samples.keys()) else None
            t = test_samples['survival_time'] if "survival_time" in test_samples.keys() else None
            k = test_samples['survival_status'] if "survival_status" in test_samples.keys() else None
            test_surv = {"c": c, "t": t, "k": k}
            self.test_surv = test_surv if None not in test_surv.values() else {}

    def embedding(self, ax, z, labels):
        """
        Plot a 2 or 3d latent embedding on axis `ax`.
            Optionally include labels, or a metric tied to each sample
        """

        # col_iterator = iter(get_cmap("tab10").colors)
        # col_iterator = iter(sns.color_palette("deep", n_colors=np.unique(labels), as_cmap=True))
        col_iterator = iter(sns.color_palette("Set2"))   # , n_colors=np.unique(labels)
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
        if z.shape[1] == 3:
            ax.set_zlabel("Embed dim $3$")

        return ax

    def histogram(self, ax, z, labels, xlabel, kde_only=True):
        """
        Plot a latent histogram on axis 'ax'.
            Optionally include labels, and (if included) plot a stacked histogram
        """
        assert len(z.shape) == 1, z.shape
        _df = pd.DataFrame({
            "latent": z,
            "Cancer type": [self.label_dict[l] for l in labels.numpy()] if self.label_dict is not None else labels
        })
        if kde_only:
            g = sns.kdeplot(data=_df, ax=ax, x="latent", palette="Set2", hue="Cancer type", legend=True, common_norm=False)
        else:
            g = sns.histplot(data=_df, ax=ax, stat="density", multiple="dodge",    # stat=count,   multiple=stack,
                             x="latent", kde=True,
                             palette="Set2", hue="Cancer type",
                             element="bars", legend=True,
                                                                                                                                                                                                                                    # kde_kws={"common_norm": True, "common_grid": True}
                             )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Kernel Density Esimation")

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
