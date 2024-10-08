from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
import logging
from SignalTransformData.survival.loader import SurvivalDataModule
from SignalTransformData.survival.generate import SurvivalDataset
from WaveLSTM.models.DeSurv import create_desurv, DeSurv

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="confs", config_name="survival_config")
def run_wavelet_desurv(cfg : DictConfig):

    torch.manual_seed(cfg.experiment.seed)

    # Make dataloader
    dm = SurvivalDataModule(**cfg.data)

    # Make model
    max_times = [np.max([b["survival_time"].max() for b in iter(loader)]) for loader in [dm.train_dataloader(),
                                                                                         dm.val_dataloader(),
                                                                                         dm.test_dataloader()
                                                                                         ]]
    model, trainer = create_desurv(data_module=dm, cfg=cfg, time_scale=np.max(max_times))

    if cfg.experiment.train:
        trainer.fit(model, datamodule=dm)
        logging.info(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        model = DeSurv.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        checkpoint = trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt"
        logging.info(f"Loading from cached checkpoint {checkpoint}")
        model = DeSurv.load_from_checkpoint(checkpoint)

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_wavelet_desurv()
