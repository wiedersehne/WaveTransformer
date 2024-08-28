from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.TCGA.data_modules.CHISEL_S0E.loaders import DataModule, Dataset
# from src.WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder, AttentiveAutoEncoder
from src.WaveTransformer.models.autoencoder import create_sa_autoencoder, AttentiveAutoEncoder

@hydra.main(version_base=None, config_path="confs", config_name="autoencoder_transformer_config")
def run_CHISEL(cfg : DictConfig):

    torch.manual_seed(cfg.experiment.seed)

    # Make dataloader
    dm = DataModule(**cfg.data)
    print(f"width {dm.W}, channels {dm.C}")
    
    # Make model
    model, trainer = create_sa_autoencoder(data_module=dm, cfg=cfg)

    if cfg.experiment.train:
        trainer.fit(model, datamodule=dm)
        logging.info(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        model = AttentiveAutoEncoder.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        checkpoint = trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt"
        logging.info(f"Loading from cached checkpoint {checkpoint}")
        model = AttentiveAutoEncoder.load_from_checkpoint(checkpoint)

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_CHISEL()
