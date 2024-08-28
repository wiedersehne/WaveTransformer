import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import logging
import numpy as np
from sklearn import preprocessing
from src.TCGA.data_modules.ascat.loaders import ASCATDataset, ASCATDataModule
from src.WaveLSTM.models.classifier import create_classifier, AttentiveClassifier
import logging
HYDRA_FULL_ERROR=1

logging.basicConfig(level=logging.INFO)
@hydra.main(version_base=None, config_path="confs", config_name="classifier_config")
def run_tcga_example(cfg : DictConfig):

    torch.manual_seed(cfg.experiment.seed)

    # Make dataloader
    dm = ASCATDataModule(**cfg.data)

    # Create model
    model, trainer = create_classifier(classes=[f"Class {i}" for i in range(1, 6)], data_module=dm, cfg=cfg)

    print(model)

    if cfg.experiment.train:
        trainer.fit(model, datamodule=dm)
        logging.info(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        # print("*******")
        model = AttentiveClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        checkpoint = trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt"
        logging.info(f"Loading from cached checkpoint {checkpoint}")
        model = AttentiveClassifier.load_from_checkpoint(checkpoint)

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_tcga_example()