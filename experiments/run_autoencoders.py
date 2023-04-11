import numpy as np
import TCGA
from SignalTransformData.data_modules.simulated import SinusoidalDataModule

from WaveLSTM.modules.encoder import create_autoencoder

# TODO: make a proper configuration (.yaml or whatever)


def run_sinusoidal_example():
    config = {
        "bias": [[-0.5, 0.5],
                 [0.5, -0.5],
                 [-0.5, 0.5],
                 [-0.5, 0.5],
                 [0, 0],
                 [0, 0]],
        "base_angular_freq": [[1 * np.pi, 2 * np.pi],
                              [1 * np.pi, 2 * np.pi],
                              [1 * np.pi, 1 * np.pi],
                              [1 * np.pi, 1 * np.pi],
                              [3 * np.pi, 2 * np.pi],
                              [3 * np.pi, 2 * np.pi]],
        "base_amplitude": [[0.5, 0.5],
                           [0.5, 0.5],
                           [0.0, 0.5],
                           [0.0, 0.5],
                           [0.5, 0.5],
                           [0.5, 0.5]],
        "transient_bool": [False, False, False, True, False, False],
        "transient_start": [[np.NaN, np.NaN],
                            [np.NaN, np.NaN],
                            [np.NaN, np.NaN],
                            [125, 250],
                            [np.NaN, np.NaN],
                            [np.NaN, np.NaN]],
        "transient_amplitude": [[np.NaN, np.NaN],
                                [np.NaN, np.NaN],
                                [np.NaN, np.NaN],
                                [0.15, -0.15],
                                [np.NaN, np.NaN],
                                [np.NaN, np.NaN]],
        "singularity_bool": [False, False, False, False, False, True],
        "singularity_start": [[np.NaN, np.NaN],
                              [np.NaN, np.NaN],
                              [np.NaN, np.NaN],
                              [np.NaN, np.NaN],
                              [np.NaN, np.NaN],
                              [50, 300]],
        "singularity_amplitude": [[np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [-0.3, 0.3]],
    }

    # Load data and filter for only the cases of interest
    dm = SinusoidalDataModule(config, samples=2000, batch_size=128, sig_length=512,
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

    train = True
    if train:
        # Train model
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")

    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/demo.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


def run_ascat_example():

    cancer_types = ['BRCA', 'OV']  # ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],  # ,  #['STAD', 'COAD'],

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.ASCATDataModule(batch_size=64, cancer_types=cancer_types,
                                                 # wgd=False,
                                                 )
    print(dm)
    print(next(iter(dm.test_dataloader())))
    print(type(next(iter(dm.test_dataloader()))))

    # Create model
    model, trainer = create_autoencoder(seq_length=dm.W, strands=2, chromosomes=23,
                                        wavelet="haar",
                                        hidden_size=512, layers=1, proj_size=0,
                                        recursion_limit=6,
                                        num_epochs=50,
                                        validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                        project="WaveLSTM-ae",
                                        run_id=f"ascat"
                                        )

    # # Train model
    # trainer.fit(model, datamodule=dm)
    # # trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    # # trained_model.freeze()
    # model.freeze()
    #
    # # Test model
    # trainer.test(model, dataloaders=dm.test_dataloader())

    train = True
    if train:
        # Train model
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")

    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/ascat.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    run_ascat_example()
    # run_sinusoidal_example()

