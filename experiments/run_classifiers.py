import torch
import TCGA
import SignalTransformData as STD

from source.model.encoder.self_attentive_encoder import create_classifier

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make .size method in DataModules so model code can be streamlined

# TODO: make a proper configuration (.yaml or whatever)


def run_sinusoidal_example(project_name):

    config = get_config(project_name)

    # Load data and filter for only the cases of interest
    dm = STD.data_modules.simulated.SinusoidalDataModule(**config)
    print(dm)

    # Create model
    model, trainer = create_classifier(num_classes=config['classes'],
                                       seq_length=config["sig_length"], strands=config["channels"], chromosomes=1,
                                       hidden_size=128, layers=1, proj_size=30,
                                       wavelet='bior4.4',
                                       coarse_skip=0,
                                       recursion_limit=None,
                                       num_epochs=5,
                                       validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                       test_hook_batch=next(iter(dm.test_dataloader())),        # TODO: update to all set
                                       project="WaveLSTM-sinusoidal-devel",
                                       run_id=f"devel"
                                       )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


def run_ascat_example(project_name):

    config = get_config(project_name)

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.ASCATDataModule(**config)
    print(dm)
    print(len(dm.train_set))

    # Create model
    model, trainer = create_classifier(num_classes=len(config["cancer_types"]), seq_length=90, strands=2, chromosomes=23,
                                       hidden_size=256, layers=1, proj_size=100,
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


def get_config(project='ascat'):
    if project == 'ascat':
        return {"batch_size": 128,
                "cancer_types": ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],
                # "wgd": False,
                }
    elif project == "sinusoidal":
        return {"batch_size": 64,
                "classes": 4,
                "samples": 2000,
                "sig_length": 256,
                "channels": 2,
                "save_to_file": "/home/ubuntu/Documents/Notebooks/wave-LSTM_benchmark/sinusoidal.csv"
                }


if __name__ == '__main__':

    run_ascat_example("ascat")
    # run_sinusoidal_example("sinusoidal")

