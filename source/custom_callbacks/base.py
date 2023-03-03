# Base class for callback classes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster


class BaseCallback(object):

    def __init__(self, val_samples=None, test_samples=None, num_samples=8):
        self.val_features = val_samples['feature'] if val_samples is not None else None
        self.val_labels = val_samples['label'] if val_samples is not None else None
        if val_samples is not None:
            self.val_sub_labels = val_samples['sub_label'] if "sub_label" in val_samples.keys() else None
        else:
            self.val_sub_labels = None

        self.test_features = test_samples['feature'] if test_samples is not None else None
        self.test_labels = test_samples['label'] if test_samples is not None else None
        if test_samples is not None:
            self.test_sub_labels = test_samples['sub_label'] if "sub_label" in test_samples.keys() else None
        else:
            self.test_sub_labels = None

        # # TODO: when updating to full val/test set, seed + sub-sample num_samples - ATM this is redundant and we just
        # # TODO: and make so None leads to no sub-batching
        # # apply callback to the batch
        # self.num_samples = num_samples
        # if val_samples is not None:
        #     assert num_samples < self.val_features.size(0)
        # if test_samples is not None:
        #     assert num_samples < self.test_features.size(0)

    @staticmethod
    def embedding(ax, z, labels=None, sub_labels=None, metric=None):
        """
        Plot a latent embedding on axis `ax`.
            Optionally include labels, or a metric tied to each sample
        """
        # standardise and scale
        z -= z.min(axis=0)
        z /= z.max(axis=0)
        scaled_metric = 0.3 * (0 + metric * 10) ** 2 if metric is not None else None

        if sub_labels is not None:
            marker_list = ["o", "$T$", "P", "X"]
            for idx, sub_cls in enumerate(np.unique(sub_labels)):
                mask = np.ma.getmask(np.ma.masked_equal(sub_labels, sub_cls))
                sm_mask = scaled_metric[mask] if scaled_metric is not None else None
                sc = ax.scatter(z[mask, 0], z[mask, 1],
                                c=labels[mask], marker=marker_list[idx], alpha=0.5, s=sm_mask)
        else:
            sc = ax.scatter(z[:, 0], z[:, 1],
                            c=labels, alpha=0.5,  s=scaled_metric)

        # Produce a legend for the classes (colors), we only want to show at most ? of them in the legend.
        # if labels is not None:
        #     kw = dict(num=len(np.unique(labels)))
        #     legend1 = ax.legend(*sc.legend_elements(**kw),
        #                         loc="upper left", title="Class", ncol=2,
        #                         bbox_to_anchor=(0.05, 1.15), fancybox=True, shadow=True,
        #                         )
        #     ax.add_artist(legend1)

        # Produce a legend for the metric (sizes), we only want to show at most 4 of them in the legend.
        # if metric is not None:
        #     kw = dict(prop="sizes", num=4, color=sc.cmap(0.7), fmt="{x:.1f}",
        #               func=lambda _s: (np.sqrt(_s / .3) - 5) / 10)
        #     legend2 = ax.legend(*sc.legend_elements(**kw),
        #                         loc="upper right", title="Count", ncol=2,
        #                         bbox_to_anchor=(0.95, 1.15), fancybox=True, shadow=True,
        #                         )

        ax.set_xlabel("Principal component $1$")
        ax.set_ylabel("Principal component $2$")

        return ax

    def heatmap(self, prediction, truth, labels, pc=None, title=""):
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
            self.embedding(ax3, pc, labels=labels, metric=np.mean(truth.reshape((truth.shape[0], -1)), axis=-1))
            ax3.set_title(f'Latent embedding')
        else:
            sns.heatmap(np.sqrt(np.abs(prediction - truth)), ax=ax3, cmap='Blues', yticklabels=labels)
            ax3.set_title(f'Root absolute error')

        fig.suptitle(title)
        plt.tight_layout()

        return fig
