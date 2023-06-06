# Create custom callbacks for our pytorch-lightning model

import numpy as np
import pickle
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

    def run_callback(self, features, labels, log_name, _trainer, _pl_module):
        # Push features through the model
        _, meta_result = _pl_module(features)
        features = np.asarray(features.detach().cpu(), dtype=np.float)
        labels = list(labels.detach().cpu().numpy())

        # Plot each resolution-embedding vectors
        wandb_images = []
        for level, hj in enumerate(meta_result['resolution_embeddings']):

            hj = np.asarray(hj.detach().cpu())

            # Project, if needed
            title = ""
            if (hj.shape[1] == 2):
                fig, ax = plt.subplots(1,1)
            elif (hj.shape[1] == 3):
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
            else:
                fig, ax = plt.subplots(1,1)
                # U-MAP
                hj = umap.UMAP(n_components=2, random_state=42).fit_transform(hj)
                title = "(U-MAP)"

            fig.suptitle(f"resolution {level + 1} embedding {title}")
            self.embedding(ax, hj, labels=labels)
            plt.tight_layout()

            wandb_images.append(wandb.Image(fig))

        # # U-MAP of concatenated resolution embeddings
        # concat_hidden = torch.concat(meta_result["hidden"], 1)
        # latent = umap.UMAP(n_components=2, random_state=42).fit_transform(np.asarray(concat_hidden.detach().cpu()))
        # fig, ax = plt.subplots(1, 1)
        # fig.suptitle(f"All resolution embeddings U-MAP")
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
                              "Val:ResolutionEmbedding", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            self.run_callback(features, self.test_labels,
                              "Test:ResolutionEmbedding", trainer, pl_module)

class SaveOutput(Callback, BaseCallback):
    """
    Callback on test epoch end to save outputs for plotting
    """
    def __init__(self, test_samples, file_path=None):
        Callback.__init__(self)
        BaseCallback.__init__(self, test_samples=test_samples)
        self.file_path = file_path if file_path is not None else "output.pkl"

    def run_callback(self, features, labels, _pl_module):
        # Push features through the model
        recon, meta_result = _pl_module(features)
        meta_result["labels"] = labels

        with open(self.file_path, 'wb') as file:
            pickle.dump(meta_result, file)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_features is not None:
            features = self.test_features.to(device=pl_module.device)
            labels = self.test_labels.to(device=pl_module.device)
            self.run_callback(features, labels, pl_module)


