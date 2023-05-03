import numpy as np
import TCGA
import torch
from SignalTransformData.data_modules.simulated import SinusoidalDataModule
from configs.demo_config import get_demo_config

from WaveLSTM.models.autoencoder import create_autoencoder

# TODO: make a proper configuration (.yaml or whatever)


def run_sinusoidal():

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(get_demo_config(), samples=2000, batch_size=128, sig_length=512,
                              save_to_file="/home/ubuntu/Documents/Notebooks/wave-LSTM_benchmark/sinusoidal.csv")
    print(dm)

    # Create model
    model, trainer = create_autoencoder(seq_length=512, strands=2, chromosomes=1,
                                        hidden_size=64, layers=1, proj_size=0,
                                        wavelet='haar',
                                        recursion_limit=7,
                                        num_epochs=50,
                                        validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                        project="WaveLSTM-ae",
                                        run_id=f"demo",
                                        )

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


def run_ascat():

    cancer_types = ['BRCA', 'OV']  # ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],  # ,  #['STAD', 'COAD'],

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=64, cancer_types=cancer_types, # wgd=False,
                                                         )
    print(dm)
    # print(next(iter(dm.test_dataloader())))
    # print(type(next(iter(dm.test_dataloader()))))

    # Create model
    model, trainer = create_autoencoder(seq_length=dm.W, strands=2, chromosomes=23,
                                        wavelet="haar",
                                        hidden_size=512, layers=1, proj_size=0, scale_embed_dim=128,
                                        recursion_limit=6,
                                        num_epochs=50,
                                        validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                        project="WaveLSTM-ae",
                                        run_id=f"ascat"
                                        )

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/ascat.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


def run_CHISEL():
    dm = TCGA.data_modules.CHISEL_S0E.loaders.DataModule(batch_size=64)

    # _X = []
    # for batch in iter(dm.train_dataloader()):
    #     _X.append(batch["feature"])
    # _X = torch.concat(_X, 0)
    # mean, std = _X.mean(0), _X.std(0)

    # print(next(iter(dm.test_dataloader()))["feature"].shape)

    # Create model
    model, trainer = create_autoencoder(seq_length=dm.W, strands=2, chromosomes=22,
                                        wavelet="haar",
                                        hidden_size=512, layers=1, proj_size=0, scale_embed_dim=128,
                                        recursion_limit=6,
                                        num_epochs=50,
                                        validation_hook_batch=next(iter(dm.val_dataloader())), # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),      # TODO: update to all set
                                        project="WaveLSTM-ae",
                                        run_id=f"CHISEL",
                                        # norm_stats = (mean, std)
                                        )

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/CHISEL.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())

if __name__ == '__main__':

    # run_ascat()
    run_sinusoidal()
    # run_CHISEL()
