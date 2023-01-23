import torch
import numpy as np
import pandas as pd
import pywt
from TCGA.data_modules.utils.helpers import get_chr_base_pair_lengths as chr_lengths


def edges2pool(subject_edge_info, equal_chr_length=False, steps=18):
    """
    Custom helper function to apply a low-pass filter on collections of (startpos, endpos).
    This used for the Haar wavelet
    """
    true_chr_lengths = chr_lengths()

    if equal_chr_length is True:
        raise NotImplementedError
    else:
        num_chrom, num_strands = 23, 2

        approximation_space = [None] * num_chrom
        # print(f"\nSubject:\n {subject_edge_info} \n==============")

        for chr in subject_edge_info["chr"].unique():
            chr_ind = 23 if chr in ['X', 'Y'] else int(chr)
            chr_length = true_chr_lengths[chr]

            subject_chr_edge = subject_edge_info[subject_edge_info["chr"] == chr].reset_index()

            # For each piecewise constant segment, i.e. (start_pos, end_pos) entry
            a_start_len = int(np.floor(chr_length / (2 ** steps)))
            a_start = np.ones((num_strands, a_start_len))
            # TODO: no padding will give edge rounding error - add padding to sequences? It would be a lot of padding...
            for idx, row in subject_chr_edge.iterrows():
                # print(f"\nRow:\n{row} of {subject_chr_edge}")

                # Positions as a percentage across average pooled length
                start_frac = row['startpos'] / chr_length
                end_frac = row['endpos'] / chr_length

                # Reduce down to bank_start approximation space (apply low pass filter `bank_start' times).
                #   In our Haar wavelet case, this is equivalent to average pooling. We calculate this using the edges
                #   as we do not want to keep the full sequence in memory.
                for strand_ind, strand in enumerate(['nMajor', 'nMinor']):
                    start_ind = int(np.floor(start_frac * a_start_len))
                    end_ind = int(np.floor(end_frac * a_start_len))
                    a_start[strand_ind, start_ind:end_ind] = row[strand]

            approximation_space[chr_ind - 1] = a_start
        # print(f"{len(approximation_space)}, {len(approximation_space[0][0])}")

        # Return list, num_chromosomes[num_strands x seq_length]
        return approximation_space


def edges2wavelet(subject_edge_info, equal_chr_length=False, pooling_steps=18, max_depth=10, wavelet='haar'):
    """
    Custom helper function to convert collections of (startpos, endpos) into the coefficients of a wavelet filter bank.
    This is passed to https://github.com/cwlgadd/TCGA/ submodule package, and called during __getitem__ to avoid storing
        many large vectors.
    """
    w = pywt.Wavelet(wavelet)

    if equal_chr_length is True:
        raise NotImplementedError
    else:
        pooling = edges2pool(subject_edge_info, equal_chr_length=equal_chr_length, steps=pooling_steps)
        pooling = np.hstack(pooling)       # Stack chromosomes

        # Get the maximum level where at least one coefficient in the output is uncorrupted by edge effects caused
        # by signal extension. Put another way, decomposition stops when the signal becomes shorter than the FIR
        # filter length for a given wavelet.
        dwt_max_level = pywt.dwt_max_level(data_len=pooling.shape[1], filter_len=w.dec_len)
        # and don't go beyond this depth, or our chosen depth if that is lower
        level = dwt_max_level if max_depth is None else np.min((max_depth, dwt_max_level))
        # Get multi-resolution wavelet filter bank
        filter_bank = pywt.wavedec(pooling, 'haar', level=level)
        print(filter_bank.shape)
    return filter_bank  # torch.tensor(pooling, dtype=torch.float), [torch.tensor(cAD, dtype=torch.float) for cAD in filter_bank]
