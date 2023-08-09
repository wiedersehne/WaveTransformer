# Ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf

import numpy as np
import TCGA
import torch
import wandb
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder
from WaveLSTM.models.DeSurv import create_desurv


def run_wavelet_desurv(dm, cancers, train=True, encoder_type="waveLSTM", J=5, r_hops=10):
    """
        train:               Train or load from checkpoint
        use_cna:

    """

    features, labels, survival_time, survival_status, days_since_birth, sex = [], [], [], [], [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
        survival_time.append(t_batch["survival_time"])
        survival_status.append(t_batch["survival_status"])
        days_since_birth.append(t_batch["days_since_birth"])
        sex += t_batch["sex"]
    test_all = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0),
                "survival_time": torch.concat(survival_time, 0),
                "survival_status": torch.concat(survival_status, 0),
                "days_since_birth": torch.concat(days_since_birth, 0),
                "sex": sex}
    print(test_all["feature"].shape)

    weight_decay= 0 #1e-5 * 2**r_hops if encoder_type.lower() == "wavelstm" else 0
    model, trainer = create_desurv(dm.label_encoder.classes_, seq_length=dm.W, channels=dm.C,
                                   J=J, r_hops=r_hops, D=1,
                                   encoder_type=encoder_type,
                                   num_epochs=200,
                                   validation_hook_batch=next(iter(dm.val_dataloader())),
                                   test_hook_batch=test_all,
                                   project="WaveLSTM-surv",
                                   run_id=f"ASCAT-attentive-deSurv",
                                   verbose=True,
                                   )

    # Normalizing stats for count number variant data
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-2] = 1
    model.normalize_stats = (mean, std)

    # Standardizing scale for survival time. Chosen empirically then fixed for all runs.
    model.time_scale = 7064   # np.max([b["survival_time"].max() for b in iter(dm.train_dataloader())])
    model.max_test_time = 5480      # np.max([b["survival_time"].max() for b in iter(dm.test_dataloader())])

    if train:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/Wave-DeSurv.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())

    # Predict
    # trainer.predict(model, dataloaders=dm.test_dataloader())

if __name__ == '__main__':

    # Load data
    cancer_types = ['THCA', 'BRCA', 'OV', 'GBM', 'HNSC']       # Filters for the cases of interest
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=256, cancer_types=cancer_types,
                                                         chrom_as_channels=True,
                                                         sampler=False)
    # print(np.unique(dm.data_frame["cancer_type"]))
    print(dm.C)

    for j in [3]:
        for r in [1]:
            print(f"J={j}, r={r}")
            run_wavelet_desurv(dm, cancer_types, J=j, r_hops=r, train=True, encoder_type="wavelstm")
