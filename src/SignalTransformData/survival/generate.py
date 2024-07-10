# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe.
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import pytorch_lightning as pl
import os
from abc import ABC
import pandas as pd
import pickle as pkl
import numpy as np
import logging
import random
import copy
from typing import Optional
try:
    from SignalTransformData.survival.simulations import SimulationModel
except:
    pass
from SignalTransformData.survival.loader import SurvivalDataset
try:
    from pysurvival.utils.display import display_baseline_simulations
except:
    logging.info("Failed to import pysurvival module. Check dependencies are fulfilled if simulation is required.")
import matplotlib.pyplot as plt


class SimulateCNA:
    """
    Create or load simulated version of count number data.
    """

    def make_insertion(self, position, width, amplitude=1.):
        """
        add an insertion which is unique to the cancer type position % of the way through.
        """
        assert position >= 0 and position <= 1
        insertion = np.floor(position * self.length * self.channels)
        t1 = (insertion % self.length).astype(int)
        channel = np.floor(insertion / self.length).astype(int)
        t2 = np.min((self.length-1, t1+width))

        self.signals[:, channel, t1:t2] += amplitude
        return

    def __init__(self, positions, channels=6, sig_length=512):
        super().__init__()
        self.positions = positions
        self.channels = channels
        self.length = sig_length
        self.N = self.positions.shape[0]
        self.n_events = self.positions.shape[1]

    def __call__(self, insertion_width=16):
        self.t = np.arange(self.length)
        self.signals = np.ones((self.N, self.channels, self.length + insertion_width))    # padded
        self.insertion_width = insertion_width

        # Convert position along sequence to index
        insertion = np.floor(self.positions * self.length * self.channels)
        # Then split into channel, and position along channel indices
        insertion_idx = (insertion % self.length).astype(int)
        insertion_chan_idx = np.floor(insertion / self.length).astype(int)

        # Add the insertions which dictate survival chance within a Cox PH model
        # TODO: vectorise
        for n in range(self.N):
            for i in range(self.n_events):
                start_pos = insertion_idx[n, i]
                channels = insertion_chan_idx[n, i]
                self.signals[n, channels, start_pos:start_pos + insertion_width] += 1

        self.signals = self.signals[:, :, :self.length]                    # Remove padding
        # self.signals_no_noise = copy.copy(self.signals)
        # self.signals += np.random.normal(0, 0.1, self.signals.shape)

    def save(self, filename="data/SimulateSinusoidal"):

        training_set = SinusoidalDataset(self.train_df, self.label_encoder)
        with open(filename + "_train.pkl", 'wb') as pfile:
            pkl.dump(training_set, pfile, pkl.HIGHEST_PROTOCOL)

        validation_set = SinusoidalDataset(self.val_df, self.label_encoder)
        with open(filename + "_val.pkl", 'wb') as pfile:
            pkl.dump(validation_set, pfile, pkl.HIGHEST_PROTOCOL)

        test_set = SinusoidalDataset(self.test_df, self.label_encoder)
        with open(filename + "_test.pkl", 'wb') as pfile:
            pkl.dump(test_set, pfile, pkl.HIGHEST_PROTOCOL)


def make_cluster(N, label,
                 sig_length=512, alpha=0.1, beta=5.0, num_base_insertions=10, plot=False, position=None):
    # For one cancer type, sample a number of CNA insertion positions, and sample the survival time given these, a
    #    sampled gender, and a normalised age.
    position_simulation = SimulationModel(survival_distribution='gompertz',
                                          risk_type='linear',
                                          censored_parameter=5.0,
                                          alpha=alpha,
                                          beta=beta)
    # Generating the survival profile for N samples belong to current CPH model (i.e. cancer type)
    feature_distributions = ["normalised_age", "gender"] + ["CNA_position"] * num_base_insertions
    surv_data = position_simulation.generate_data(feature_distributions, num_samples=N)
    positions = surv_data.to_numpy()[:, 2:-2]

    cna_simulation = SimulateCNA(positions, channels=2, sig_length=sig_length)
    cna_simulation(insertion_width=int(sig_length / 8))

    # Add Cox PH model specific insertion
    if position is not None:
        cna_simulation.make_insertion(position=position[0], width=int(sig_length * position[1]), amplitude=1)

    if plot:
        plt.imshow(cna_simulation.signals.reshape((N, -1)))
        plt.colorbar()
        display_baseline_simulations(position_simulation, figure_size=(5, 5))
        plt.show()

    d = {
        'feature': [cna_simulation.signals[i, :] for i in range(cna_simulation.N)],
        'label': [label for _ in range(cna_simulation.N)],
        'survival_time': surv_data.to_numpy()[:, -2],
        'survival_status': surv_data.to_numpy()[:, -1],
        'age': surv_data.to_numpy()[:, 0],
        'sex': ["male" if surv_data.to_numpy()[i, 1] == 0 else "female" for i in range(cna_simulation.N)]
    }
    return pd.DataFrame(data=d)

def generate_datasets(N, alphas, betas, positions, save=False, **kwargs):
    frames = []
    for label, (alpha, beta, position) in enumerate(zip(alphas, betas, positions)):
        frame = make_cluster(N=N,
                             label=label,
                             alpha=alpha, beta=beta, position=position,
                             **kwargs)
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True, axis=0)

    # Create label encoder
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(frame.label.unique())

    # Split frame into training, validation, and test
    train_df, test_df = sk_split(frame, test_size=0.2)
    test_df, val_df = sk_split(frame, test_size=0.5)

    if save:

        training_set = SurvivalDataset(train_df, label_encoder)
        with open("data/SimulateCNA_train.pkl", 'wb') as pfile:
            pkl.dump(training_set, pfile, pkl.HIGHEST_PROTOCOL)

        validation_set = SurvivalDataset(val_df, label_encoder)
        with open("data/SimulateCNA_val.pkl", 'wb') as pfile:
            pkl.dump(validation_set, pfile, pkl.HIGHEST_PROTOCOL)

        test_set = SurvivalDataset(test_df, label_encoder)
        with open("data/SimulateCNA_test.pkl", 'wb') as pfile:
            pkl.dump(test_set, pfile, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    # Harder version
    # alphas = [0.5, 1]
    # betas = [1, 1]
    # Easier version
    alphas = [0.1, 1.5]
    betas = [1, 1]

    positions = [None, (0.2, 1/64)]  # (0.2, 1/32)
    # widths = [8/64, 1/64]

    generate_datasets(1000, alphas, betas, positions=positions,
                      num_base_insertions=30,
                      save=True, plot=True)

    with open('data/SimulateCNA_train.pkl', 'rb') as file:
        train_dataset = pkl.load(file)

        for batch in train_dataset:
            for key in batch.keys():
                print(f"{key}".ljust(30) + f"{batch[key].shape}")
            break
