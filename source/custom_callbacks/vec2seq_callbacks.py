# Create custom callbacks for the vec2seq pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


class LatentSpace(Callback):
    """
    Callback to view latent embedding of labelled data,
     plotting the first two principal components of the latent embedding
    """
    def __init__(self, val_samples=None, test_samples=None, label_dict=None):
        super().__init__()
        self.val_features = val_samples['feature'] if val_samples is not None else None
        self.val_labels = val_samples['label'] if val_samples is not None else None

        self.test_features = test_samples['feature'] if test_samples is not None else None
        self.test_labels = test_samples['label'] if test_samples is not None else None

        self.label_dict = label_dict

    def embedding(self, x, labels, metric, eigenvalues, title=""):

        # standardise
        x -= x.min(axis=0)
        x /= x.max(axis=0)
        s = 0.3*(0+metric*10)**2

        fig, (ax, ax_hist) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 1]})
        fig.suptitle(title)

        sc = ax.scatter(x[:, 0], x[:, 1], s=s, c=labels, alpha=0.5)

        # Produce a legend for the classes (colors), we only want to show at most ? of them in the legend.
        if self.label_dict is not None:
            kw = dict(num=len(np.unique(labels)), color=sc.cmap(0.7), fmt="Class {x}",
                      func=lambda l: (np.sqrt(s / .3) - 0) / 10)
        else:
            kw = dict(num=len(np.unique(labels)))

        # legend1 = ax.legend(*sc.legend_elements(**kw),
        #                     loc="upper left", title="Class", ncol=2,
        #                     bbox_to_anchor=(0.05, 1.15), fancybox=True, shadow=True,
        #                     )
        # ax.add_artist(legend1)

        # Produce a legend for the metric (sizes), we only want to show at most 4 of them in the legend.
        kw = dict(prop="sizes", num=4, color=sc.cmap(0.7), fmt="{x:.1f}",
                  func=lambda _s: (np.sqrt(_s / .3)-5) / 10)
        legend2 = ax.legend(*sc.legend_elements(**kw),
                            loc="upper right", title="Count", ncol=2,
                            bbox_to_anchor=(0.95, 1.15), fancybox=True, shadow=True,
                            )

        ax.set_xlabel("Principal component $1$")
        ax.set_ylabel("Principal component $2$")

        ax_hist.barh(np.arange(eigenvalues.shape[0]) + 1, eigenvalues, orientation='horizontal')
        ax_hist.set_xlabel("EV")

        plt.tight_layout()
        return fig

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, teacher_forcing):
        # Push features through the model
        _, meta_result = _pl_module(features, teacher_forcing=teacher_forcing)

        features = np.asarray(features.detach().cpu(), dtype=np.float)
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)

        metric = np.mean(features.reshape((features.shape[0], -1)), axis=-1)
        wandb_images = []

        # Hidden embedding vectors
        for level, z in enumerate(meta_result['hidden']):
            z = z.detach().cpu()
            # Concatenate hidden vectors in each direction
            z = torch.concat([z[i, :, :] for i in range(z.shape[0])], dim=-1)
            # Plot first two principal components of the latent space
            (U, S, V) = torch.pca_lowrank(z, niter=2)
            _pc = np.asarray(torch.matmul(z, V[:, :2]))
            _ev = S**2/(z.shape[0]-1)
            # print(f"h: z shape {z.shape} -> {_ev.shape}")

            wandb_images.append(
                wandb.Image(self.embedding(_pc, labels, metric, _ev,
                                           title=f"Hidden LSTM embedding, layer {level + 1}",
                                           )
                            )
            )

        _trainer.logger.experiment.log({
            log_name: wandb_images
        })
        plt.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.val_features.to(device=pl_module.device)
            labels = self.val_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Val:EmbeddingPCA", trainer, pl_module, 0.)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test:EmbeddingPCA", trainer, pl_module, 0.)


class FeatureSpace1d(Callback):
    """
    Callback on validation epoch end, plotting predictions, as heatmap and as individual line plots
    """
    def __init__(self, val_samples, test_samples, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_features = val_samples['feature'] if val_samples is not None else None
        self.val_labels = val_samples['label'] if val_samples is not None else None
        self.test_features = test_samples['feature'] if test_samples is not None else None
        self.test_labels = test_samples['label'] if test_samples is not None else None

        if val_samples is not None:
            assert num_samples < self.val_features.size(0)
        if test_samples is not None:
            assert num_samples < self.test_features.size(0)

        # TODO
        # : when updating to full val/test set, seed + sub-sample num_samples

    @staticmethod
    def run_callback(features, labels, log_name, _trainer, _pl_module, evaluation=True):
        # If we're testing/validating model then we do not use teacher forcing
        teacher_forcing = 0. if evaluation is True else evaluation
        # Push features through the model
        recon, meta_result = _pl_module(features, teacher_forcing=teacher_forcing)

        recon = np.asarray(recon.detach().cpu())
        features = np.asarray(features.detach().cpu())
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)

        # Log results
        _trainer.logger.experiment.log({
            log_name:
                [wandb.Image(heatmap(recon[:, strand, :, :].reshape(features.shape[0], -1),
                                     features[:, strand, :, :].reshape(features.shape[0], -1),
                                     labels, title=f"Strand index {strand}"))
                 for strand in range(features.shape[1])]
        })
        plt.close('all')

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     if self.val_features is not None:
    #         features = self.val_features.to(device=pl_module.device)
    #         labels = self.val_labels.to(device=pl_module.device)
    #         self.run_callback(features, labels, "Validation_PredictionHeatmap", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test_PredictionHeatmap", trainer, pl_module)


class RecurrentFeatureSpace1d(Callback):
    """
    Callback on validation epoch end, plotting predictions, as heatmap and as individual line plots
    """
    def __init__(self, val_samples, test_samples, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_features = val_samples['feature'] if val_samples is not None else None
        self.val_labels = val_samples['label'] if val_samples is not None else None
        self.test_features = test_samples['feature'] if test_samples is not None else None
        self.test_labels = test_samples['label'] if test_samples is not None else None

        if val_samples is not None:
            assert num_samples < self.val_features.size(0)
        if test_samples is not None:
            assert num_samples < self.test_features.size(0)

        # TODO
        # : when updating to full val/test set, seed + sub-sample num_samples

    @staticmethod
    def run_callback(features, labels, log_name, _trainer, _pl_module, evaluation=True):
        # If we're testing/validating model then we do not use teacher forcing
        teacher_forcing = 0. if evaluation is True else evaluation
        # Push features through the model
        _, meta_result = _pl_module(features, teacher_forcing=teacher_forcing)

        recurrent_recon = [np.asarray(x.detach().cpu()) for x in meta_result['pred_recurrent_recon']]
        true_recurrent_recon = [np.asarray(x.detach().cpu()) for x in meta_result['true_recurrent_recon']]
        labels = np.asarray(labels.detach().cpu(), dtype=np.int)

        # TODO: temporarily removing second strand as its just a copy
        recurrent_recon = [x[:, 0, :, :] for x in recurrent_recon]
        true_recurrent_recon = [x[:, 0, :, :] for x in true_recurrent_recon]

        _trainer.logger.experiment.log({
            log_name:
                [wandb.Image(heatmap(recon_rnn.reshape(features.shape[0], -1),
                                     true_rnn.reshape(features.shape[0], -1),
                                     labels, title=f"Depth index {depth} (2nd strand not shown)"))
                 for depth, (recon_rnn, true_rnn) in enumerate(zip(recurrent_recon, true_recurrent_recon))]
        })
        plt.close('all')

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     if self.val_features is not None:
    #         features = self.val_features.to(device=pl_module.device)
    #         labels = self.val_labels.to(device=pl_module.device)
    #         self.run_callback(features, labels, "Val_PredictionHeatmapRecurrent", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.val_features is not None:
            features = self.test_features.to(device=pl_module.device)
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, "Test_PredictionHeatmapRecurrent", trainer, pl_module)


def heatmap(prediction, truth, labels, title):
    permute_idx = np.argsort(labels)
    prediction = prediction[permute_idx, :]
    truth = truth[permute_idx, :]
    labels = labels[permute_idx]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12, 18))
    vmax = np.min((5, np.max((truth.max(), prediction.max()))))
    vmin = np.min((1, np.min((truth.min(), prediction.min()))))

    sns.heatmap(prediction, ax=ax1, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
    sns.heatmap(truth, ax=ax2, cmap='Blues', vmin=vmin, vmax=vmax, yticklabels=labels)
    sns.heatmap(np.log(np.abs(prediction - truth)), ax=ax3, cmap='Blues', yticklabels=labels)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Locus")
        ax.set_ylabel("Sample label (permuted to order)")
        ax.xaxis.set_tick_params(rotation=90)
        for idx, label in enumerate(ax.yaxis.get_ticklabels()):
            if idx % 5 != 0:
                label.set_visible(False)

    ax1.set_title(f'Prediction')
    ax2.set_title(f'Truth')
    ax3.set_title(f'Log absolute error')
    fig.suptitle(title)
    plt.tight_layout()
    return fig
