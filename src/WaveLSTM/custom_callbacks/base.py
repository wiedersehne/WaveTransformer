# Base class for callback classes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import scipy.cluster.hierarchy as hcluster


class BaseCallback(object):

    def __init__(self, val_samples=None, test_samples=None, label_dict=None):
        self.val_features = val_samples['feature'] if val_samples is not None else None
        self.val_labels = val_samples['label'] if val_samples is not None else None

        self.test_features = test_samples['feature'] if test_samples is not None else None
        self.test_labels = test_samples['label'] if test_samples is not None else None

        self.label_dict = label_dict

    def embedding(self, ax, z, labels, metric=None):
        """
        Plot a latent embedding on axis `ax`.
            Optionally include labels, or a metric tied to each sample
        """
        # standardise and scale
        z -= z.min(axis=0)
        z /= z.max(axis=0)
        if metric is not None:
            metric = (((metric - np.min(metric)) / np.max(metric)) * 20) + 5
            # metric *= metric

        col_iterator = iter(get_cmap("tab10").colors)
        for lbl in np.unique(labels):
            mask = np.ma.getmask(np.ma.masked_equal(labels, lbl))
            color = next(col_iterator)
            c = self.label_dict[lbl] if self.label_dict is not None else lbl
            ax.scatter(z[mask, 0], z[mask, 1], c=np.array([color]), s=metric[mask] if metric is not None else None,
                       label=c, alpha=0.5, edgecolors='none')

        ax.legend()
        # print(*sc.legend_elements())

        # # produce a legend with the unique colors from the scatter
        # legend1 = ax.legend(*sc.legend_elements(), loc="lower left", title="Classes")
        # ax.add_artist(legend1)

        # if metric is not None:
        #     # produce a legend with a cross section of sizes from the scatter
        #     handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
        #     print(handles, labels)
        #     legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

        ax.set_xlabel("t-SNE $1$")
        ax.set_ylabel("t-SNE $2$")

        return ax

    def heatmap(self, prediction, truth, labels, pc=None, scale=False, title=""):
        """
        Create a subfigure of truth, prediction.
            Options: if pc is None then in 3rd figure we plot the root absolute error, else we plot the pc and order by
                     1st pc
        """
        if pc is None:
            permute_idx = np.argsort(labels)
        else:
            # permute_idx = np.argsort(pc[:, 0] + pc[:, 1])
            clusters = hcluster.fclusterdata(pc, 0.10, criterion="distance")
            permute_idx = np.argsort(clusters)

        prediction = prediction[permute_idx, :]
        truth = truth[permute_idx, :]
        pc = pc[permute_idx, :] if pc is not None else None
        labels = labels[permute_idx]

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))

        # Plot prediction and truth
        vmax = np.min((5, np.max((truth.max(), prediction.max()))))
        vmin = np.min((1, np.min((truth.min(), prediction.min()))))
        sns.heatmap(prediction, ax=ax1, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax1.set_title(f'Prediction')
        sns.heatmap(truth, ax=ax2, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
        ax2.set_title(f'Truth')

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("Locus")
            ax.set_ylabel(f"Sample (permuted by {'label' if pc is None else 'hierarchical latent clustering'})")
            ax.xaxis.set_tick_params(rotation=90)
            for idx, label in enumerate(ax.yaxis.get_ticklabels()):
                if idx % 5 != 0:
                    label.set_visible(False)

        if pc is not None:
            metric = np.mean(truth.reshape((truth.shape[0], -1)), axis=-1) if scale else None
            self.embedding(ax3, pc, labels=labels, metric=metric)
            ax3.set_title(f'Latent embedding')
        else:
            sns.heatmap(np.sqrt(np.abs(prediction - truth)), ax=ax3, cmap='Blues', yticklabels=labels)
            ax3.set_title(f'Root absolute error')

        fig.suptitle(title)
        plt.tight_layout()

        return fig
