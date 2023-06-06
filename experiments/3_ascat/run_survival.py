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
                                           hidden_size=128, layers=1, proj_size=0, scale_embed_dim=2, recursion_limit=5,
                                           decoder="fc", r_hops=1, nfc=64,
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
    torch.save(model.a_encoder.state_dict(), "configs/ASCAT-attentive-pretrain.pt")

    wandb.finish()

def run_wavelet_desurv(dm, train=True, use_cna=True):

    model, trainer = create_desurv(seq_length=dm.W, channels=dm.C,
                                   wavelet="haar",
                                   hidden_size=128, layers=1, proj_size=0, scale_embed_dim=2, recursion_limit=5,
                                   use_CNA=use_cna, pre_trained="configs/ASCAT-attentive-pretrain.pt",
                                   num_epochs=30,
                                   validation_hook_batch=next(iter(dm.val_dataloader())),
                                   test_hook_batch=next(iter(dm.test_dataloader())),
                                   project="WaveLSTM-surv",
                                   run_id=f"ASCAT-attentive-deSurv",
                                   verbose=True
                                   )

    # Normalizing stats for count number variant data
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-2] = 1
    model.normalize_stats = (mean, std)

    # Standardizing scale for survival time
    # t_train_max = np.max([b["survival_time"].max() for b in iter(dm.train_dataloader())])
    t_test_max = np.max([b["survival_time"].max() for b in iter(dm.test_dataloader())])
    # model.norm_time = t_train_max
    model.max_test_time = t_test_max

    if train:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/Wave-DeSurv.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':


    # Filters for the cases of interest
    cancer_types = ["BRCA"]  #"['OV', 'BRCA', 'GBM', 'KIRC', 'HNSC', 'LGG']

    # Load data
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=256, cancer_types=cancer_types,
                                                         chrom_as_channels=False,
                                                         sampler=False)

    print(dm.C)

    pre_train_encoder(dm, train=True)
    run_wavelet_desurv(dm, train=True, use_cna=True)