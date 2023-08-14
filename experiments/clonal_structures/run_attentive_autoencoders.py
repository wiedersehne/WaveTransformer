import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import TCGA
import torch
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder


def make_dataloaders(cfg_data):
    # Load data
    dm = TCGA.data_modules.CHISEL_S0E.loaders.DataModule(**cfg_data)

        # stack batches to test set samples on test hook
    features, labels = [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
    test_all = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0)}

    return dm, next(iter(dm.val_dataloader())), test_all


@hydra.main(version_base=None, config_path="confs", config_name="autoencoder_config")
def run_CHISEL(cfg : DictConfig):
    """
    """
    # print(OmegaConf.to_yaml(cfg))

    #################
    # Make dataloader
    #################
    dm, val_data, test_data = make_dataloaders(cfg.data)
    print(f"width {dm.W}, channels {dm.C}")

    ############
    # Make model
    ############
    pool_targets = False
    model, trainer = create_sa_autoencoder(data_module=dm, test_data=test_data, val_data=val_data, cfg=cfg,
                                           pool_targets=pool_targets)

    # Normalizing stats
    if pool_targets is False:
        features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], dim=0)
        mean = torch.mean(features, 0)
        std = torch.std(features, 0)
        std[std < 1e-4] = 1
        model.normalize_stats = (mean, std)

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_CHISEL()
