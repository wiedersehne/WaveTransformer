import TCGA
import torch
from WaveLSTM.models.autoencoder import create_autoencoder

def run_ascat():

    cancer_types = ['BRCA', 'OV']  # ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],  # ,  #['STAD', 'COAD'],

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=64, cancer_types=cancer_types, # wgd=False,
                                                         )
    print(dm)
    # print(next(iter(dm.test_dataloader())))
    # print(type(next(iter(dm.test_dataloader()))))

    # Create model
    model, trainer = create_autoencoder(seq_length=dm.W, channels=2*23,
                                        wavelet="haar",
                                        hidden_size=512, layers=1, proj_size=0, scale_embed_dim=128,
                                        recursion_limit=6,
                                        num_epochs=50,
                                        validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                        project="WaveLSTM-chisel",
                                        run_id=f"ascat-ae",
                                        )

    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std<1e-2] = 1
    model.normalize_stats = (mean, std)

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/ascat.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_ascat()
