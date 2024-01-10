import hydra
from omegaconf import DictConfig, OmegaConf
from SignalTransformData.simulated import SinusoidalDataModule
from demo_config import get_demo_config
from WaveLSTM.models.classifier import create_classifier
import torch

@hydra.main(version_base=None, config_path="confs", config_name="classifier_config")
def run_sinusoidal_example(cfg : DictConfig):
    """
    """
    # print(OmegaConf.to_yaml(cfg))

    #################
    # Make dataloader
    #################
    dm = SinusoidalDataModule(get_demo_config(), **cfg.data)
    print(f"width {dm.W}, channels {dm.C}")

    # Create model
    model, trainer = create_classifier(classes=[f"Class {i}" for i in range(1, 7)], data_module=dm, cfg=cfg)
    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    model.normalize_stats = (torch.mean(features, 0), torch.std(features, 0))

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_sinusoidal_example()
