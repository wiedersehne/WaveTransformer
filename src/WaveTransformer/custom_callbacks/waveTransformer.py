# Create custom callbacks for our pytorch-lightning model

import numpy as np
import pickle

import sklearn.manifold
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from WaveLSTM.custom_callbacks.base import BaseCallback


class ResolutionEmbedding(Callback, BaseCallback):
    """
    Callback to view latent embedding  of labelled data at each recurrent step,
     plotting the first two principal components of each latent embedding, and the free-energy of each component
    """
    def __init__(self, val_samples=None, test_samples=None, label_dictionary=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, val_samples=val_samples, test_samples=test_samples, label_dict=label_dictionary)

    def run_callback(self, features, labels, log_name, _trainer, _pl_module, proj="tsne", proj_3d=True, **kwargs):

        # Push features through the model
        _, meta_result = _pl_module(features, **kwargs)

        # Plot each resolution-embedding vectors
        wandb_images = []
        for level, hj in enumerate(meta_result['resolution_embeddings']):

            hj = np.asarray(hj)

            print(hj.shape)

            # Plot depends on shape of latent dimension
            if hj.shape[1] == 1:
                # Histogram for 1-d embedding
                fig, ax = plt.subplots(1, 1)
                fig.suptitle(f"j={level+1}")
                self.histogram(ax, hj[:, 0], labels=labels, xlabel=f"Latent resolution embedding $(h_j)$")
            elif (hj.shape[1] == 2):
                # 2d scatter plot for 2-d embedding
                fig, ax = plt.subplots(1, 1)
                fig.suptitle(f"j={level+1} resolution embedding")
                self.embedding(ax, hj, labels=labels)
            elif (hj.shape[1] == 3) and (proj_3d is False):
                # 3d scatter plot for 3-d embedding
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                fig.suptitle(f"j-{level+1} resolution embedding")
                self.embedding(ax, hj, labels=labels)
            else:
                # Else we project with U-MAP to 2-d and do a 2d scatter
                fig, ax = plt.subplots(1, 1)
                if proj=="umap":
                    fig.suptitle(f"j-{level+1} resolution embedding (U-MAP)")
                    hj_proj = umap.UMAP(n_components=2).fit_transform(hj)  # random_state=42
                else:
                    perp = np.max((3, np.min((30, int(0.1 * hj.shape[0])))))
                    fig.suptitle(f"j-{level + 1} resolution embedding (t-SNE, perplexity={perp})")
                    hj_proj = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perp).fit_transform(hj)
                self.embedding(ax, hj_proj, labels=labels)

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
            self.run_callback(features, self.val_labels, "Val:Resolution", trainer, pl_module, **val_surv)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, "Test:Resolution", trainer, pl_module, **test_surv)

class SaveOutput(Callback, BaseCallback):
    """
    Callback on test epoch end to save outputs for plotting
    """
    def __init__(self, test_samples, file_path=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, test_samples=test_samples)
        self.file_path = file_path if file_path is not None else "output.pkl"

    def run_callback(self, features, labels, _pl_module, **kwargs):
        # Push features through the model
        recon, meta_result = _pl_module(features, **kwargs)
        meta_result["labels"] = labels

        with open(self.file_path, 'wb') as file:
            pickle.dump(meta_result, file)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            # Send to device
            features = self.test_features.to(device=pl_module.device)
            test_surv = {k: v.to(device=pl_module.device, non_blocking=True) for k, v in
                         self.test_surv.items()}  # possibly empty surv dictionary
            # Run callback
            self.run_callback(features, self.test_labels, pl_module, **test_surv)
