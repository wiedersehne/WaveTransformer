import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
from abc import ABC
from experiments.configs.config import extern

pl.seed_everything(42)

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = FILE_PATH + r'/CNdata/ascat/ReleasedData/TCGA_SNP6_hg19'

# Used for transforming true sequences into padded, equal length sequences.
PAD_LENGTH = 1.002
CHROM_LENGTHS = {"1": np.floor(249250621 * PAD_LENGTH),
                 "2": np.floor(243199373 * PAD_LENGTH),
                 "3": np.floor(198022430 * PAD_LENGTH),
                 "4": np.floor(191154276 * PAD_LENGTH),
                 "5": np.floor(180915260 * PAD_LENGTH),
                 "6": np.floor(171115067 * PAD_LENGTH),
                 "7": np.floor(159138663 * PAD_LENGTH),
                 "8": np.floor(146364022 * PAD_LENGTH),
                 "9": np.floor(141213431 * PAD_LENGTH),
                 "10": np.floor(135534747 * PAD_LENGTH),
                 "11": np.floor(135006516 * PAD_LENGTH),
                 "12": np.floor(133851895 * PAD_LENGTH),
                 "13": np.floor(115169878 * PAD_LENGTH),
                 "14": np.floor(107349540 * PAD_LENGTH),
                 "15": np.floor(102531392 * PAD_LENGTH),
                 "16": np.floor(90354753 * PAD_LENGTH),
                 "17": np.floor(81195210 * PAD_LENGTH),
                 "18": np.floor(78077248 * PAD_LENGTH),
                 "19": np.floor(59128983 * PAD_LENGTH),
                 "20": np.floor(63025520 * PAD_LENGTH),
                 "21": np.floor(48129895 * PAD_LENGTH),
                 "22": np.floor(51304566 * PAD_LENGTH),
                 "X": np.floor(155270560 * PAD_LENGTH),
                 "Y": np.floor(59373566 * PAD_LENGTH)
                 }


def load_frame(length=500, file_path=None):
    """
    Load ASCAT released count number data.

    :param length: How long the down-sampled vector should be.
    :param file_path: Where to load/save data frame relative to this file.
    :return: A dataframe which will be used in the count number data loader.

    """
    # Load frame from given file_path
    if file_path is not None:
        try:
            return pd.read_pickle(FILE_PATH + file_path)
        except FileNotFoundError:
            print(f"Could not load pickled dataframe from path {FILE_PATH + file_path}, creating new...")

    # Load in the label information -> patient ID, cancer type, WGD, gi
    sample_label_frame = pd.read_csv(DATA_PATH + r'/TCGA.giScores.wgd.txt', delimiter='\t')
    print(f"Value counts for each cancer class: {sample_label_frame.cancer_type.value_counts()}")

    # Load in the count number data
    # Loop over segments files (each file belongs to one patient)
    all_frames = []
    for idx_seg, entry in tqdm(enumerate(os.scandir(DATA_PATH + '/segments/')),
                               desc='Parsing count number data from files',
                               total=12240):

        if (entry.path.endswith(r".segments.txt")) and entry.is_file():
            try:
                # Extract patient identifier from segments file, cross reference against label file, and store labels
                subject_id = re.search(r'segments/(.+?).segments.txt', entry.path).group(1)
                subject_labels = sample_label_frame[sample_label_frame['patient'] == subject_id]
                subject_labels = subject_labels[['cancer_type', 'wgd', 'gi']].to_numpy()[0]

                # Load segment file
                sample_frame = pd.read_csv(entry.path, sep='\t')

                # For each chromosome create windowed CN vectors, normalised so all have 'length' bins.
                # Do not round to integers yet. Here there is a large risk of coarse graining out finer details.
                for chromosome in sample_frame.chr.unique():
                    scale = length / CHROM_LENGTHS[chromosome]
                    sample_frame.loc[sample_frame.chr == chromosome, 'startpos'] *= scale
                    sample_frame.loc[sample_frame.chr == chromosome, 'endpos'] *= scale

                sample_frame.insert(6, 'cancer_type', subject_labels[0])
                sample_frame.insert(7, 'wgd', subject_labels[1])
                sample_frame.insert(8, 'gi', subject_labels[2])

                all_frames.append(sample_frame)

            except Exception as e:
                # Catch empty files
                # print(f"Error {e.__class__} occurred.")
                pass

    frame = pd.concat(all_frames, ignore_index=True)

    # Put into memory efficient formatting
    # size_before = frame.memory_usage(deep=True).sum()
    frame["nmajor"] = pd.to_numeric(frame["nMajor"], downcast="unsigned")
    frame["nminor"] = pd.to_numeric(frame["nMinor"], downcast="unsigned")
    frame["gi"] = pd.to_numeric(frame["gi"], downcast="unsigned")
    frame["startpos"] = pd.to_numeric(frame["startpos"], downcast="unsigned")
    frame["endpos"] = pd.to_numeric(frame["endpos"], downcast="unsigned")
    frame["cancer_type"] = frame["cancer_type"].astype("category")
    frame["chr"] = frame["chr"].astype("category")
    frame["wgd"] = frame["wgd"].astype("category")
    # print(f'Achieved compression of {frame.memory_usage(deep=True).sum() / size_before}')

    # TODO: report how many higher fidelity regions were lost from coarse graining.

    # Save frame to file_path
    if file_path is not None:
        pd.to_pickle(frame, FILE_PATH + file_path)

    return frame


def filter_frame(data_frame, cancer_types=None, wgd=None, chromosomes=None):
    """

    :param data_frame:       Frame we want to filter
    :param cancer_types:     list of strings, where each string is the cancer type
    :param wgd:
    :param chromosomes:
    :return:
    """
    # Apply feature of interest filters
    if cancer_types is not None:
        mask = data_frame.cancer_type.isin(cancer_types)
        data_frame = data_frame[mask]
    if wgd is not None:
        data_frame = data_frame[data_frame['wgd'] == wgd]
    if chromosomes is not None:
        raise NotImplementedError('Bugged feature')         # TODO
        mask = data_frame.chr.isin(chromosomes)
        data_frame = data_frame[mask]
    assert len(data_frame.index) > 0, 'There are no samples with these filter criterion'

    # Encode remaining cancer type labels, so they can be used by the model later
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(data_frame.cancer_type.unique())

    #

    # Split frame into training, validation, and test
    unique_samples = data_frame['sample'].unique()
    train_labels, test_labels = sk_split(unique_samples, test_size=0.2)
    test_labels, val_labels = sk_split(test_labels, test_size=0.5)
    train_df = data_frame[data_frame['sample'].isin(train_labels)]
    val_df = data_frame[data_frame['sample'].isin(val_labels)]
    test_df = data_frame[data_frame['sample'].isin(test_labels)]
    assert len(train_df.cancer_type.unique()) == len(data_frame.cancer_type.unique())

    # Random sampler weights
    weight_dict = {}
    for cancer_id, group in train_df.groupby('cancer_type'):
        unique_samples = len(group['sample'].unique()) / len(train_df['sample'].unique())
        if unique_samples > 0:
            weight_dict[cancer_id] = 1 / unique_samples

    # Report any class (im)balance
    df_dict = {'train frame': train_df, 'validation frame': val_df, 'test frame': test_df}
    for key in df_dict:
        assert len(df_dict[key].cancer_type.unique()) == len(cancer_types), f'{key} lost class representation'
        for cancer_id, group in df_dict[key].groupby('cancer_type'):
            unique_samples = len(group['sample'].unique())
            if unique_samples > 0:
                print(f'In {key} there are {unique_samples} samples with cancer type {cancer_id}')

    return data_frame, (train_df, val_df, test_df), weight_dict, label_encoder


class DatasetBDLSTM(Dataset):
    """
    From dataframe create count_number data pipeline feeder, formatted for BDLSTM model.
    We keep data in the condensed (startpos, endpos) format for memory efficiency at the cost of some minor overhead.
    """

    def get_series(self, sequence):
        """
        Helper function to convert collections of (startpos, endpos) into full (likely coarse grained) sequences. This
        is called during __getitem__ to avoid storing many large vectors.
        """
        nmajor, nminor = np.ones((self.length,)), np.ones((self.length,))
        start_pos = sequence['startpos'].to_numpy()
        end_pos = sequence['endpos'].to_numpy()

        for idx, (start, end) in enumerate(zip(start_pos, end_pos)):
            # Get start and end points for indexing. If last segment, always round up 'endpos' due to added padding zone
            start = int(round(start))
            if idx == start_pos.shape[0] - 1:
                end = int(np.ceil(end))
            else:
                end = int(round(end))

            nmajor[start:end] = sequence.nmajor.to_numpy()[idx]
            nminor[start:end] = sequence.nminor.to_numpy()[idx]

        return nmajor, nminor

    def __init__(self, data: pd.DataFrame, label_encoder, weight_dict: dict = None):
        """
        """
        # TODO: transform on samples needed?

        self.data_frame = data
        self.label_encoder = label_encoder

        # sequence length
        # TODO: move this so its shared for train/val/test frames, or some other way to make it more robust
        self.length = int(np.ceil(self.data_frame['endpos'].max()))

        # Extract and format count number data from data frame. Separate by chromosome for potential model channels
        # TODO: This can't be vectorised as we need each sample's chromosome sequence to have its own structure,
        #  ... but can make indexing/code structure here much more logical and clean!
        chrom_sequences = []
        for chromosome_id, group_c in self.data_frame.groupby('chr'):

            sequences = []                      # List of feature series and labels for one chromosome
            weights = []
            for patient_id, group_cp in group_c.groupby('sample'):

                # The label information
                cancer_type = group_cp['cancer_type'].to_numpy()[0]
                cancer_type_enc = list(self.label_encoder.classes_).index(cancer_type)
                label = [cancer_type_enc,
                         group_cp['wgd'].to_numpy()[0],
                         group_cp['gi'].to_numpy()[0],
                         ]
                if weight_dict is not None:
                    weights.append(weight_dict[cancer_type])

                # The information required to reconstruct the sequence of count numbers
                sequence_features = group_cp[['startpos', 'endpos', 'nmajor', 'nminor']]

                # Store for __getitem__
                sequences.append((sequence_features, label))

            chrom_sequences.append(sequences)

        # The sequences along each chromosome
        self.sequences = chrom_sequences         # List of chromosomes[list of samples[tuple of features and label]]
        self.weights = weights

    def __len__(self):
        # All chromosomes are transformed to have the same length
        return len(self.sequences[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Collect sample
        chr_seq_maj, chr_seq_minor = [], []
        for chrom in range(len(self.sequences)):
            sequence, label = self.sequences[chrom][idx]
            # Convert (startpos, endpos) format into full sequence
            sequence_major, sequence_minor = self.get_series(sequence)
            chr_seq_maj.append(sequence_major)
            chr_seq_minor.append(sequence_minor)

        seq_maj = np.stack(chr_seq_maj, axis=1)
        seq_min = np.stack(chr_seq_minor, axis=1)

        return dict(
            nmajor=torch.Tensor(seq_maj),             # Sequence 'length' x Number of chromosomes
            nminor=torch.Tensor(seq_min),             # Sequence 'length' x Number of chromosomes
            cancer_type=torch.Tensor([label[0]]),     # Scalar
            wgd=torch.Tensor([label[1]]),
            gi=torch.Tensor([label[2]])
        )


class DataModuleBDLSTM(pl.LightningDataModule, ABC):
    """

    """
    def __init__(self, train_df, val_df, test_df, label_encoder, batch_size=128, weight_dict: dict = None):
        super(DataModuleBDLSTM, self).__init__()
        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df
        self.label_encoder = label_encoder
        self.n_classes = len(label_encoder.classes_)
        self.weight_dict = weight_dict
        self.batch_size = batch_size

    def train_dataloader(self):
        training_set = DatasetBDLSTM(self.train_df,
                                     self.label_encoder,
                                     weight_dict=self.weight_dict
                                     )
        if self.weight_dict is not None:
            sampler = WeightedRandomSampler(training_set.weights, len(training_set.weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            sampler=sampler,
            dataset=training_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=DatasetBDLSTM(self.val_df,
                                  self.label_encoder,
                                  ),
            batch_size=self.batch_size,
            num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=DatasetBDLSTM(self.test_df,
                                  self.label_encoder,
                                  ),
            batch_size=self.batch_size,
            num_workers=os.cpu_count()
        )


@extern
def cn_pipeline_constructor(length=250, file_path=None, cancer_types=None, wgd=None, chromosomes=None, batch_size=256):
    # Load data and filter for only the cases of interest
    df = load_frame(length=length, file_path=file_path)

    _, (df_train, df_val, df_test), train_weights, label_encoder = filter_frame(df,
                                                                                cancer_types=cancer_types,
                                                                                wgd=wgd,
                                                                                chromosomes=chromosomes)

    data_module = DataModuleBDLSTM(df_train, df_val, df_test,
                                   weight_dict=train_weights,
                                   label_encoder=label_encoder,
                                   batch_size=batch_size)
    data_module.setup()

    return data_module


def main_test():
    data_module = cn_pipeline_constructor()

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch test index {batch_idx}')
            print(f"cancer type counts {torch.unique(batch['cancer_type'], return_counts=True)}")
            print(batch['nmajor'].shape)
            print(batch['wgd'].shape)


if __name__ == '__main__':

    main_test()
