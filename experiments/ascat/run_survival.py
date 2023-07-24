# Ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf

import numpy as np
import TCGA
import torch
import wandb
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder
from WaveLSTM.models.DeSurv import create_desurv

def pre_train_encoder(dm, train=True):

    features, labels = [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
    test_all = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0)}

    # Create modelon_validation_epoch_end
    model, trainer = create_sa_autoencoder(seq_length=dm.W, channels=dm.C,
                                           wavelet="haar",
                                           hidden_size=32, layers=1, proj_size=0, scale_embed_dim=1, recursion_limit=5,
                                           decoder="fc", r_hops=1, nfc=256,
                                           num_epochs=30,
                                           validation_hook_batch=next(iter(dm.val_dataloader())),
                                           test_hook_batch=test_all,
                                           project="WaveLSTM-surv",
                                           run_id=f"ASCAT-attentive-pretrain",
                                           )

    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-4] = 1
    model.normalize_stats = (mean, std)

    if train:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/ASCAT-attentive-pretrain.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())

    # Dump attentie auto-encoder's state to file so it can be used as a pre-trained model for survival example
    torch.save(model.a_encoder.state_dict(), "logs/ASCAT-attentive-pretrain.pt")

    wandb.finish()

def run_wavelet_desurv(dm, cancers, train=True, encoder_type="waveLSTM", pre_train=None):
    """
        train:               Train or load from checkpoint
        use_cna:
        pre_train:           whether to use an encoder pre-trained on another task. Default: No pre-training

    """

    features, labels, survival_time, survival_status, days_since_birth, sex = [], [], [], [], [], []
    for t_batch in iter(dm.test_dataloader()):
        print(t_batch.keys())
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

    model, trainer = create_desurv(dm.label_encoder.classes_, seq_length=dm.W, channels=dm.C,
                                   wavelet="haar",
                                   hidden_size=16, layers=1, proj_size=1, scale_embed_dim=1, recursion_limit=4,
                                   atn_drop=0.0, r_hops=10,
                                   surv_dropout=0.,
                                   encoder_type=encoder_type, pre_trained=pre_train,
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

    # Standardizing scale for survival time
    t_train_max = np.max([b["survival_time"].max() for b in iter(dm.train_dataloader())])
    t_test_max = np.max([b["survival_time"].max() for b in iter(dm.test_dataloader())])
    model.time_scale = t_train_max
    model.max_test_time = t_test_max
    print(f"Norm time {model.time_scale}, {model.max_test_time}")

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

    # pre_train_encoder(dm, train=True)
    run_wavelet_desurv(dm, cancer_types, train=True, encoder_type="lstm")