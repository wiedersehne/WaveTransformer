import torch
import TCGA
import SignalTransformData as STD

from source.vec2seq import create_vec2seq
from source.model.decoder.rnn_wavelet import WaveletLSTM
from source.model.decoder.rnn_lstmwavelet import WaveletConv1dLSTM
from source.model.encoder.sequence_encoder import SequenceEncoder

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

    seq_length = config["sig_length"]
    chromosomes = 1
    strands = config["channels"]

    # Create decoder
    wave_lstm = WaveletLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                            hidden_size=256, layers=1, bidirectional=True, proj_size=50)
    wave_convlstm = WaveletConv1dLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                                      hidden_size=256, layers=1, proj_size=50)

    # Create model
    model, trainer = create_vec2seq(recurrent_net=wave_convlstm,
                                    wavelet='bior4.4',  #  'coif4'
                                    coarse_skip=0,
                                    recursion_limit=None,
                                    auto_reccurent=False,
                                    num_epochs=25,
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

    seq_length = 90
    chromosomes = 23
    strands = 2

    # Create decoder
    wave_lstm = WaveletLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                            hidden_size=256, layers=1, bidirectional=True, proj_size=50)
    wave_convlstm = WaveletConv1dLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                                      hidden_size=256, layers=1, proj_size=50)

    # Create model
    model, trainer = create_vec2seq(recurrent_net=wave_convlstm,
                                    wavelet='haar',  # 'bior4.4',  # 'coif4'
                                    coarse_skip=0,
                                    recursion_limit=10,
                                    auto_reccurent=False,
                                    num_epochs=100,
                                    validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                    test_hook_batch=next(iter(dm.test_dataloader())),        # TODO: update to all set
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
                "classes": 10,
                "samples": 2000,
                "sig_length": 256,
                "channels": 2,
                "save_to_file": "/home/ubuntu/Documents/Notebooks/wave-LSTM_benchmark/sinusoidal.csv"
                }


if __name__ == '__main__':

    run_ascat_example("ascat")
    # run_sinusoidal_example("sinusoidal")

