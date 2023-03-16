import numpy as np
import TCGA
from SignalTransformData.data_modules.simulated import SinusoidalDataModule

from source.model.encoder.encoder import create_autoencoder

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
                                [-0.15, 0.15],
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
    dm = SinusoidalDataModule(config, samples=2000, batch_size=64, sig_length=512,
                              save_to_file="/home/ubuntu/Documents/Notebooks/wave-LSTM_benchmark/sinusoidal.csv")
    print(dm)

    # Create model
    model, trainer = create_autoencoder(seq_length=512, strands=2, chromosomes=1,
                                        hidden_size=64, layers=1, proj_size=20,
                                        wavelet='haar',  #bior4.4',  #  'coif4'
                                        coarse_skip=0,
                                        recursion_limit=None,
                                        num_epochs=25,
                                        validation_hook_batch=next(iter(dm.val_dataloader())), # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),      # TODO: update to all set
                                        project="WaveLSTM-sinusoidal-devel",
                                        run_id=f"devel"
                                        )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


def run_ascat_example():

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.ASCATDataModule(batch_size=128, cancer_types=['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],
                                                 # wgd=False,
                                                 )
    print(dm)
    print(len(dm.train_set))

    # Create model
    model, trainer = create_autoencoder(seq_length=90, strands=2, chromosomes=23,
                                        hidden_size=256, layers=1, proj_size=0,
                                        coarse_skip=0, recursion_limit=10,
                                        num_epochs=100,
                                        validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                        test_hook_batch=next(iter(dm.test_dataloader())),       # TODO: update to all set
                                        project="WaveLSTM-ASCAT-devel",
                                        run_id=f"devel"
                                        )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    # run_ascat_example()
    run_sinusoidal_example()

