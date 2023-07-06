import TCGA
import torch
from WaveLSTM.models.autoencoder import create_autoencoder

def run_CHISEL():
    dm = TCGA.data_modules.CHISEL_S0E.loaders.DataModule(batch_size=64)

    # Merge val batches for validation hooks
    # features, labels = [], []
    # for batch in iter(dm.val_dataloader()):
    #     features.append(batch["feature"])
    #     labels.append(batch["label"])
    # val_all = {"feature": torch.concat(features, 0),
    #             "label": torch.concat(labels, 0)}
    # Merge test batches for testing hooks
    features, labels = [], []
    for batch in iter(dm.test_dataloader()):
        features.append(batch["feature"])
        labels.append(batch["label"])
    test_all = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0)}

    # Create model
    model, trainer = create_autoencoder(seq_length=dm.W, channels=2*22,
                                        wavelet="haar",
                                        hidden_size=256, layers=1, proj_size=0, scale_embed_dim=3,
                                        recursion_limit=6,
                                        num_epochs=50,
                                        validation_hook_batch=next(iter(dm.val_dataloader())),
                                        test_hook_batch=test_all,
                                        project="WaveLSTM-chisel",
                                        run_id=f"chisel-ae",
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
