import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import TCGA
import torch
import wandb
import pytorch_lightning as pl
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder
from WaveLSTM.models.DeSurv import create_desurv

# pl.seed_everything(42)


def make_dataloaders(config):
    # Load data
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(**config.data)
    # print(np.unique(dm.data_frame["cancer_type"]))

    # validation data for validation hooks
    val_data = next(iter(dm.val_dataloader()))

    # Stack all test data for test hook
    features, labels, survival_time, survival_status, days_since_birth, sex = [], [], [], [], [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
        survival_time.append(t_batch["survival_time"])
        survival_status.append(t_batch["survival_status"])
        days_since_birth.append(t_batch["days_since_birth"])
        sex += t_batch["sex"]
    test_data = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0),
                "survival_time": torch.concat(survival_time, 0),
                "survival_status": torch.concat(survival_status, 0),
                "days_since_birth": torch.concat(days_since_birth, 0),
                "sex": sex}
    # print(test_all["feature"].shape)

    return dm, val_data, test_data


@hydra.main(version_base=None, config_path="confs", config_name="survival_config")
def run_wavelet_desurv(cfg : DictConfig):
    """
    """
    # print(OmegaConf.to_yaml(cfg))
    # torch.manual_seed(cfg.experiment.seed)

    #################
    # Make dataloader
    #################
    dm, val_data, test_data = make_dataloaders(cfg)

    ############
    # Make model
    ############
    model, trainer = create_desurv(data_module=dm, test_data=test_data, val_data=val_data, cfg=cfg)

    # Normalizing stats for count number alteration data
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], dim=0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-2] = 1
    model.normalize_stats = (mean, std)
    model.time_scale =  7064 # np.max([b["survival_time"].max() for b in iter(dm.train_dataloader())]) #
    # Range of times we evaluate test metrics on. Chosen empirically, then fixed for all runs.
    model.max_test_time =  5480   # np.max([b["survival_time"].max() for b in iter(dm.test_dataloader())])     #

    # Report range of observed times for each data split
    max_times = [np.max([b["survival_time"].max() for b in iter(loader)]) for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]]
    print(max_times)

    if cfg.experiment.train:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())

    # Predict
    # trainer.predict(model, dataloaders=dm.test_dataloader())

if __name__ == '__main__':

            run_wavelet_desurv()
