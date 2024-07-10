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
from SignalTransformData.sinusoidal.generate import SinusoidalDataset

class SinusoidalDataModule(pl.LightningDataModule):
    """
    """

    training_set = None
    validation_set = None
    test_set = None

    @property
    def num_cancer_types(self):
        return len(self.label_encoder.classes_)

    def __init__(self, batch_size=128):
        r"""
        """
        super().__init__()

        self.batch_size = batch_size
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if stage == "fit" or stage is None:
            with open(f'{dir_path}/data/SimulateSinusoidal_train.pkl', 'rb') as pfile:
                self.training_set = pkl.load(pfile)

            with open(f'{dir_path}/data/SimulateSinusoidal_val.pkl', 'rb') as pfile:
                self.validation_set = pkl.load(pfile)

        if stage == "test" or stage is None:
            with open(f'{dir_path}/data/SimulateSinusoidal_test.pkl', 'rb') as pfile:
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

    with open('data/SimulateSinusoidal_train.pkl', 'rb') as pfile:
        train_dataset = pkl.load(pfile)

    dm = SinusoidalDataModule(batch_size=256)
    dm.setup()

    for batch in dm.train_dataloader():
        # print(batch.keys())
        print(batch["feature"].shape)
        print(batch["label"].shape)
