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


class SimulateSinusoidal:
    """
    Create or load simulated sinusoidal version of count number data.
    """

    @staticmethod
    def damped_signal(t, initial_amplitude, decay_rate, angular_frequency, init_phase_angle):
        return initial_amplitude * np.exp(-decay_rate * t) * np.cos(angular_frequency * t - init_phase_angle)

    @staticmethod
    def make_singularity(x, t1, width, amplitude=1.):
        """
        Make singularity structure in a batch of signals
        x: batched signal (Batch size, signal length)
        """
        x[:, t1:t1+width] -= amplitude
        x[:, t1+width:t1+2*width] += amplitude
        return x

    def make_transient(self, x, initial_amplitude, decay_rate, angular_frequency, init_phase_angle, t1, t2):
        """
        Make a transient structure in a batch of signals
        x: batched signal (Batch size, signal length)
        """
        sig_t = np.arange(t1, t2)
        x[:, sig_t] += self.damped_signal(sig_t / t2, initial_amplitude, decay_rate, angular_frequency, init_phase_angle)
        return x

    def __init__(self, 
                 config,
                 discretise:     bool=True
                ):
        super().__init__()
        self.config = config
        self.discretise = discretise
        
        self.classes = len(config["bias"])
        self.channels = len(config["bias"][0])

    def __call__(self, samples=2000, sig_length=512):
        
        self.n = samples
        self.length = sig_length
        self.t = np.arange(self.length)

        # And randomly assign each sample to a different class with equal probability
        self.labels = np.random.choice([i for i in range(self.classes)],
                                       size=self.n, p=[1./self.classes for _ in range(self.classes)])
        self.signals = np.zeros((self.n, self.channels, self.length))

        singularity_width = 8    # int(np.floor(np.log2(self.length)))
        transient_width = 64      # int(np.floor(np.log2(self.length)))

        for cls in range(self.classes):
            mask = (np.ma.getmask(np.ma.masked_equal(self.labels, cls)))

            # Add bias
            bias = self.config["bias"][cls]
            for channel in range(self.channels):
                self.signals[mask, channel, :] += bias[channel]

            # Add base signal
            base_angular_freq = self.config["base_angular_freq"][cls]
            base_amplitude = self.config["base_amplitude"][cls]
            base_init_phase_angle = 0
            for channel in range(self.channels):
                self.signals[mask, channel, :] += \
                    base_amplitude[channel] * np.cos((base_angular_freq[channel] * self.t / self.length)
                                                     - base_init_phase_angle)

            # Add transient signal
            if self.config["transient_bool"][cls] is True:
                s_t1 = self.config["transient_start"][cls]
                # s_t2 = [i + transient_width for i in s_t1]
                s_amp = self.config["transient_amplitude"][cls]
                for channel in range(self.channels):
                    self.signals[mask, channel, :] = self.make_singularity(self.signals[mask, channel, :],
                                                                           s_t1[channel], transient_width,
                                                                           amplitude=s_amp[channel])
            
            # Add singularity signal
            if self.config["singularity_bool"][cls] is True:
                s_t1 = self.config["singularity_start"][cls]
                s_amp = self.config["singularity_amplitude"][cls]
                for channel in range(self.channels):
                    self.signals[mask, channel, :] = self.make_singularity(self.signals[mask, channel, :],
                                                                           s_t1[channel], singularity_width,
                                                                           amplitude=s_amp[channel])

        if self.discretise is True:
            self.signals = (((self.signals*0.99) + 1) * 2).astype(int)
        
        self.signals_no_noise = copy.copy(self.signals)
        
        if self.discretise is True:
            self.signals += np.random.choice([-1,0,1], self.signals.shape, p=[0.05, 0.9, 0.05])
        else:
            self.signals += np.random.normal(0, 0.1, self.signals.shape)

        d = {'CNA_noisefree': [self.signals_no_noise[i, :] for i in range(self.n)],
             'CNA': [self.signals[i, :] for i in range(self.n)],
             'labels': self.labels,
             }
        self.frame = pd.DataFrame(data=d)

        # Encode remaining type labels, so they can be used by the model later
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(self.frame.labels.unique())

        # Split frame into training, validation, and test
        self.train_df, test_df = sk_split(self.frame, test_size=0.2)
        self.test_df, self.val_df = sk_split(self.frame, test_size=0.2)

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


def demo_config():

    config = {
            "bias": [[-0.5, 0.5],
                     [0.5, -0.5],
                     [-0.5, 0.5],
                     [-0.5, 0.5],
                     [0, 0],
                     [0, 0]],
            "base_angular_freq": [[1 * np.pi, 2 * np.pi],
                                  [1 * np.pi, 2 * np.pi],
                                  [1 * np.pi, 1 * np.pi],
                                  [1 * np.pi, 1 * np.pi],
                                  [3 * np.pi, 2 * np.pi],
                                  [3 * np.pi, 2 * np.pi]],
            "base_amplitude": [[0.5, 0.5],
                               [0.5, 0.5],
                               [0.0, 0.5],
                               [0.0, 0.5],
                               [0.5, 0.5],
                               [0.5, 0.5]],
            "transient_bool": [False, False, False, True, False, False],
            "transient_start": [[np.NaN, np.NaN],
                                [np.NaN, np.NaN],
                                [np.NaN, np.NaN],
                                [125, 275],
                                [np.NaN, np.NaN],
                                [np.NaN, np.NaN]],
            "transient_amplitude": [[np.NaN, np.NaN],
                                    [np.NaN, np.NaN],
                                    [np.NaN, np.NaN],
                                    [0.5, -0.5],
                                    [np.NaN, np.NaN],
                                    [np.NaN, np.NaN]],
            "singularity_bool": [False, False, False, False, False, True],
            "singularity_start": [[np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [50, 300]],
            "singularity_amplitude": [[np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [-0.5, 0.5]],
        }
    return config


class SinusoidalDataset(Dataset):

    def __init__(self, 
                 data:           pd.DataFrame, 
                 label_encoder,
                ):
        """
        """
        self.data_frame = data
        self.label_encoder = label_encoder
        self.n = len(self.data_frame.index)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features
        sample = self.data_frame.loc[self.data_frame.index[idx]]
        feature = sample['CNA']

        # Get label
        label = self.data_frame.loc[self.data_frame.index[idx], ['labels']][0]
        label_enc = list(self.label_encoder.classes_).index(label)

        batch = {"CNA": torch.tensor(feature, dtype=torch.float),
                 "label": torch.tensor(label_enc),
                 }
        return batch

def make_dataset():
    generator = SimulateSinusoidal(demo_config())
    generator(samples=2000, sig_length=512)
    generator.save()

if __name__ == "__main__":

    

    with open('data/SimulateSinusoidal_train.pkl', 'rb') as file:
        train_dataset = pkl.load(file)

    # for batch in train_dataset:
    #     print(batch)
