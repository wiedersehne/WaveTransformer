import numpy as np
import torch
import TCGA
from SignalTransformData.data_modules.simulated import SinusoidalDataModule
from configs.demo_config import get_demo_config

from WaveLSTM.models.classifier import create_classifier


# TODO: make a proper configuration (.yaml or whatever)


def run_sinusoidal_example():

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(get_demo_config(), samples=2000, batch_size=256, sig_length=512,
                              save_to_file="/home/ubuntu/Documents/Notebooks/wave-LSTM_benchmark/sinusoidal.csv")
    print(dm)

    features, labels = [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
    test_batch = {"feature": torch.concat(features, 0),
                  "label": torch.concat(labels, 0)}
    print(test_batch)

    # Create model
    model, trainer = create_classifier(classes=[f"Class {i}" for i in range(6)],
                                       seq_length=512, strands=2, chromosomes=1,
                                       hidden_size=32, layers=1, proj_size=0, scale_embed_dim=64,
                                       recursion_limit=7,
                                       validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                       test_hook_batch=test_batch,
                                       run_id=f"demo",
                                       verbose=True
                                       )

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo.ckpt")

    # Test model
    trainer.test(model, dataloaders=dm.test_dataloader())


def run_ascat_example():

    cancer_types = ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG']  # ,  #['STAD', 'COAD'],  ['BRCA', 'OV']

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=256, cancer_types=cancer_types)
    print(dm)

    # Create model
    model, trainer = create_classifier(classes=cancer_types, seq_length=dm.W, strands=2, chromosomes=23,
                                       hidden_size=32, layers=1, proj_size=0, scale_embed_dim=128,
                                       recursion_limit=5,
                                       validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                       test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                       run_id=f"ascat"
                                       )

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/ascat.ckpt")

    # Test model
    trainer.test(model, test_dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    # run_ascat_example()
    run_sinusoidal_example()

