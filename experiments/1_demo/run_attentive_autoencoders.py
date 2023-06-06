from SignalTransformData.data_modules.simulated import SinusoidalDataModule
from demo_config import get_demo_config
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder
import torch

def run_sinusoidal():

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(get_demo_config(), samples=2000, batch_size=128, sig_length=512)
    print(dm)

    # Create model
    model, trainer = create_sa_autoencoder(seq_length=512, channels=2,
                                           hidden_size=8, layers=1, proj_size=0, scale_embed_dim=2,
                                           r_hops=1, decoder="fc", decoder_width=512,
                                           wavelet='haar',
                                           recursion_limit=7,
                                           num_epochs=100,
                                           validation_hook_batch=next(iter(dm.val_dataloader())),
                                           test_hook_batch=next(iter(dm.test_dataloader())),
                                           project="WaveLSTM-demo",
                                           run_id=f"demo-attentive-ae",
                                           )
    # Normalizing stats
    # features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    # model.normalize_stats = (torch.mean(features, 0), torch.std(features, 0))

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo_attentive_autoencoder.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_sinusoidal()
