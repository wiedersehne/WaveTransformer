from SignalTransformData.data_modules.simulated import SinusoidalDataModule
from demo_config import get_demo_config
from WaveLSTM.models.classifier import create_classifier
import torch


def run_sinusoidal_example():

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(get_demo_config(), samples=2000, batch_size=256, sig_length=512)
    print(dm)

    # Create model
    model, trainer = create_classifier(classes=[f"Class {i}" for i in range(1, 7)],
                                       seq_length=512, channels=2,
                                       hidden_size=32, layers=1, proj_size=0, scale_embed_dim=2,
                                       clf_nfc=256,
                                       recursion_limit=7,
                                       validation_hook_batch=next(iter(dm.val_dataloader())),
                                       test_hook_batch=next(iter(dm.test_dataloader())),
                                       project="WaveLSTM-demo",
                                       run_id=f"demo-attentive-clf",
                                       )

    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    model.normalize_stats = (torch.mean(features, 0), torch.std(features, 0))

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo_attentive_classifier.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_sinusoidal_example()
