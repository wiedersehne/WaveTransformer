from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import logging
from SignalTransformData.sinusoidal.loader import SinusoidalDataModule
from SignalTransformData.sinusoidal.generate import SinusoidalDataset
from WaveLSTM.models.classifier import create_classifier, AttentiveClassifier

logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="confs", config_name="classifier_config")
def run_sinusoidal_example(cfg : DictConfig):

    torch.manual_seed(cfg.experiment.seed)

    # Make dataloader
    dm = SinusoidalDataModule(**cfg.data)

    # Create model
    model, trainer = create_classifier(classes=[f"Class {i}" for i in range(1, 7)], data_module=dm, cfg=cfg)

    if cfg.experiment.train:
        trainer.fit(model, datamodule=dm)
        logging.info(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        model = AttentiveClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        checkpoint = trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt"
        logging.info(f"Loading from cached checkpoint {checkpoint}")
        model = AttentiveClassifier.load_from_checkpoint(checkpoint)

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_sinusoidal_example()
