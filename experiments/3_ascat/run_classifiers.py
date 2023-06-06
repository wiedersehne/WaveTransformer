import numpy as np
import torch
import TCGA
from WaveLSTM.models.classifier import create_classifier



def run_sinusoidal_example():

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(get_demo_config(), samples=2000, batch_size=256, sig_length=512)
    print(dm)

    # Create model
    model, trainer = create_classifier(classes=[f"Class {i}" for i in range(6)],
                                       seq_length=512, channels=2,
                                       hidden_size=128, layers=1, proj_size=0, scale_embed_dim=2,
                                       clf_nfc=256,
                                       recursion_limit=7,
                                       validation_hook_batch=next(iter(dm.val_dataloader())),
                                       test_hook_batch=next(iter(dm.test_dataloader())),
                                       run_id=f"demo",
                                       verbose=True
                                       )

    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    model.normalize_stats = (torch.mean(features, 0), torch.std(features, 0))

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


def run_ascat_example():

    cancer_types = ['OV', 'BRCA', 'GBM', 'KIRC', 'HNSC', 'LGG']  # ,  #['STAD', 'COAD'],  ['BRCA', 'OV']

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=256, cancer_types=cancer_types, sampler=False)
    print(dm)

    # Create model
    model, trainer = create_classifier(classes=cancer_types,
                                       seq_length=dm.W, channels=2*23,
                                       hidden_size=512, layers=1, proj_size=0, scale_embed_dim=2,
                                       recursion_limit=4,
                                       validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                       test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                       run_id=f"ascat"
                                       )

    # Normalizing stats
    features = torch.concat([batch["feature"].reshape((-1, 46, dm.W)) for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-2] = 1
    model.normalize_stats = (mean, std)

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/ascat.ckpt")

    # Test model
    trainer.test(model, test_dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_sinusoidal_example()
    # run_ascat_example()

