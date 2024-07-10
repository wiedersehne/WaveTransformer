# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe.
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import pytorch_lightning as pl
import os
from abc import ABC
import pandas as pd
import numpy as np
import logging
import random
import copy
from typing import Optional
import pickle as pkl


class SurvivalDataset(Dataset):

    def __init__(self, data: pd.DataFrame, label_encoder):
        """
        """
        self.data_frame = data
        self.n = len(self.data_frame.index)
        self.label_encoder = label_encoder

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # subject
        subject = self.data_frame.loc[self.data_frame.index[idx]]

        age = torch.tensor(subject["age"])
        gender = torch.tensor(1 if subject["sex"] == "male" else 0)
        baseline_covariates = torch.stack((age, gender), dim=0)

        # Get features
        return {'covariates': baseline_covariates.float(),
                'CNA': torch.tensor(subject["feature"]).float(),
                'survival_time': torch.tensor(subject["survival_time"]).float(),
                'survival_status': torch.tensor(subject["survival_status"]).float(),
                'label': torch.tensor(subject["label"]).float(),
                }

class SurvivalDataModule(pl.LightningDataModule):
    """
    """

    training_set = None
    validation_set = None
    test_set = None

    @property
    def num_cancer_types(self):
        raise NotImplementedError
        # return len(self.label_encoder.classes_)


    def __init__(self, batch_size=128):
        r"""
        """
        super().__init__()

        self.batch_size = batch_size
        self.setup()

        self.labels = [i for i in range(4)]
        self.W  = 512
        self.C = 2

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if stage == "fit" or stage is None:
            with open(f'{dir_path}/data/SimulateCNA_train.pkl', 'rb') as pfile:
                self.training_set = pkl.load(pfile)
                self.label_encoder = self.training_set.label_encoder

            with open(f'{dir_path}/data/SimulateCNA_val.pkl', 'rb') as pfile:
                self.validation_set = pkl.load(pfile)

        if stage == "test" or stage is None:
            with open(f'{dir_path}/data/SimulateCNA_test.pkl', 'rb') as pfile:
                self.test_set = pkl.load(pfile)

    def train_dataloader(self):
        return DataLoader(
            self.training_set,
            sampler=None,
            batch_size=self.batch_size,
            num_workers=np.min((os.cpu_count(),8)),
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=np.min((os.cpu_count(),8)),
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=np.min((os.cpu_count(),8)),
            shuffle=False
        )


if __name__ == "__main__":

    # with open('data/SimulateCNA_train.pkl', 'rb') as pfile:
    #     train_dataset = pkl.load(pfile)

    dm = SurvivalDataModule(batch_size=256)
    dm.setup()

    for batch in dm.train_dataloader():
        for key in batch.keys():
            print(f"{key}".ljust(30) + f"{batch[key].shape}")
        break

    import matplotlib.pyplot as plt
    for label in np.unique(batch["label"]):
        times_k = batch["survival_time"][batch["label"] == label]
        plt.hist(times_k, label=label)
    plt.legend()
    plt.show()
