from SignalTransformData.data_modules.simulated import SinusoidalDataModule
from demo_config import get_demo_config
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder
import torch

def run_sinusoidal():

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(get_demo_config(), samples=2000, batch_size=32, sig_length=512)
    print(dm)

    # Combine batches for validation and test hooks
    hook_batches = []
    for loader in [dm.val_dataloader(), dm.test_dataloader()]:
        features, labels = [], []
        for batch in iter(loader):
            features.append(batch["feature"])
            labels.append(batch["label"])
        hook_batches.append({"feature": torch.concat(features, 0),
                             "label": torch.concat(labels, 0)}
                            )

    # Create model
    model, trainer = create_sa_autoencoder(seq_length=512, channels=2,
                                           hidden_size=32, layers=1, proj_size=3, scale_embed_dim=10,
                                           r_hops=1, decoder="rccae",
                                           wavelet='haar',
                                           recursion_limit=7,
                                           num_epochs=100,
                                           validation_hook_batch=hook_batches[0],
                                           test_hook_batch=hook_batches[1],
                                           project="WaveLSTM-demo",
                                           run_id=f"demo-attentive-ae",
                                           )
    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    model.normalize_stats = (torch.mean(features, 0), torch.std(features, 0))

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo_attentive_autoencoder.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_sinusoidal()
