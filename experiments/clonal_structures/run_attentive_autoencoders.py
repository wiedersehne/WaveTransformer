import numpy as np
import TCGA
import torch
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder

# TODO: make a proper configuration (.yaml or whatever)

def run_CHISEL():
    dm = TCGA.data_modules.CHISEL_S0E.loaders.DataModule(batch_size=32, sampler=False, chr_length=256)

    features, labels = [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
    test_all = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0)}

    # Create modelon_validation_epoch_end
    model, trainer = create_sa_autoencoder(seq_length=dm.W, channels=2*22,
                                           wavelet="haar",
                                           hidden_size=256, layers=1, proj_size=0, scale_embed_dim=3,
                                           r_hops=1,  decoder="rccae", pool_targets=False,
                                           recursion_limit=4,
                                           num_epochs=75,
                                           validation_hook_batch=next(iter(dm.val_dataloader())),
                                           test_hook_batch=test_all,
                                           project="WaveLSTM-chisel",
                                           run_id=f"chisel-attentive-ae",
                                           )

    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-4] = 1
    model.normalize_stats = (mean, std)

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/CHISEL.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_CHISEL()
