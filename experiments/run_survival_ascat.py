# Ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf
import numpy as np
import pandas as pd
import torch
from pycox.datasets import support
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader
from DeSurv.src.classes import ODESurvSingle
import TCGA
import matplotlib.pyplot as plt

from WaveLSTM.models.DeSurv import create_desurv

def run_desurv_ascat(load=False):
    cancer_types = ['BRCA', 'OV', 'GBM'] # ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG']  # ,  #['STAD', 'COAD'],

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.ASCATDataModule(batch_size=64, cancer_types=cancer_types)
    print(dm)
    print(next(iter(dm.test_dataloader())).keys())

    # t_train_max = np.max([b["survival_time"].max() for b in iter(dm.train_dataloader())])
    # t_n_test_min = np.min([b["survival_time"].min() / t_train_max for b in iter(dm.test_dataloader())])
    # t_n_test_max = np.max([b["survival_time"].max() / t_train_max for b in iter(dm.test_dataloader())])
    model, trainer = create_desurv(seq_length=dm.W, strands=2, chromosomes=23,
                                   wavelet="haar",
                                   hidden_size=512, layers=1, proj_size=0, scale_embed_dim=128,
                                   recursion_limit=6,
                                   num_epochs=50,
                                   validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                   test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                   project="WaveLSTM-aeSurv",
                                   run_id=f"ascat",
                                   verbose=True
                                   )
    if not load:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/DeSurv_demo-v2.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())

    # Inference

    pred_dicts = trainer.predict(model, dataloaders=dm.test_dataloader())
    surv = pd.concat([p_dict["surv"] for p_dict in pred_dicts], ignore_index=True, axis=1)
    t_test = np.concatenate([p_dict["t_test"] for p_dict in pred_dicts])
    e_test = np.concatenate([p_dict["e_test"] for p_dict in pred_dicts])
    lbl_test = np.concatenate([p_dict["lbl_test"] for p_dict in pred_dicts])

    # Evaluate
    ev = EvalSurv(surv, t_test, e_test, censor_surv='km')
    time_grid = np.linspace(t_n_test_min.min(), 0.9, 1000)
    print(f"C_td: {ev.concordance_td()}")
    print(f"IBS: {ev.integrated_brier_score(time_grid)}")
    print(f"NBLL: {ev.integrated_nbll(time_grid)}")

    # Plot KM curves
    surv_np = surv.to_numpy()
    print(surv_np.shape)
    cols = ['k', 'b', 'g']
    fig, axes = plt.subplots(1, len(np.unique(lbl_test)))
    for i, ax in enumerate(axes):
        idx_i = np.where(lbl_test == i)[0]
        ax.plot(surv.index * t_train_max, surv_np[:, idx_i], c=cols[i], alpha=0.2, label=cancer_types[i])
    plt.show()

def run_wavesurv_ascat():
    raise NotImplementedError


if __name__ == '__main__':

    run_desurv_ascat()