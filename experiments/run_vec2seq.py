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


def run_simulated_example(project_name):

    config = get_config(project_name)

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.simulated.MarkovDataModule(**config)
    # print(dm)

    latent_dimension = config["classes"]
    seq_length = config["length"]
    chromosomes = 1
    strands = 2

    # Create encoder
    encoder = SequenceEncoder(in_features=seq_length, out_features=latent_dimension, n_hidden=128, n_layers=3,
                              dropout=0.6, bidirectional=True, in_channels=2, out_channels=2,
                              kernel_size=3, stride=5, padding=1)

    # Create decoder
    decoder = WaveletLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                          hidden_size=256, layers=2, bidirectional=True, proj_size=30)

    # Create model
    auto_reccurent = False
    wandb_name = project_name + f"_auto{auto_reccurent}_hdim{decoder.hidden_size}_haar"
    model, trainer = create_vec2seq(encoder_model=encoder, decoder_model=decoder,
                                    wavelet='haar',  # 'bior4.4',  # 'coif4'
                                    coarse_skip=0, recursion_limit=None,
                                    auto_reccurent=auto_reccurent, teacher_forcing_ratio=0.5,
                                    run_id=wandb_name,
                                    num_epochs=15,
                                    validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                    test_hook_batch=next(iter(dm.test_dataloader()))  # TODO: update to all set
                                    )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


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
                                      hidden_size=256, layers=1)

    # Create model
    model, trainer = create_vec2seq(recurrent_net=wave_convlstm,
                                    wavelet='coif4', # 'bior4.4',  #'coif4'
                                    coarse_skip=0,
                                    recursion_limit=10,
                                    auto_reccurent=False,
                                    num_epochs=100,
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

    seq_length = 128
    chromosomes = 23
    strands = 2

    # Create decoder
    wave_lstm = WaveletLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                            hidden_size=256, layers=1, bidirectional=True, proj_size=50)
    wave_convlstm = WaveletConv1dLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                                      hidden_size=256, layers=1)

    # Create model
    model, trainer = create_vec2seq(recurrent_net=wave_lstm,
                                    wavelet='haar',  # 'bior4.4',  # 'coif4'
                                    coarse_skip=0,
                                    recursion_limit=10,
                                    auto_reccurent=False,
                                    num_epochs=100,
                                    validation_hook_batch=next(iter(dm.val_dataloader())),  # TODO: update to all set
                                    test_hook_batch=next(iter(dm.test_dataloader())),        # TODO: update to all set
                                    project="WaveLSTM-ASCAT-devel",
                                    run_id=f"LSTM-concatall"
                                    )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


def get_config(project='ascat'):
    if project == 'Quiet':
        return {"steps": 2,
                "classes": 10,
                "n_class_bases": 4,
                "n_bases_shared": 0,
                "length": 1000,
                "n": 10000,
                }
    elif project == 'Unstable':
        return {"steps": 5,
                "classes": 10,
                "n_class_bases": 30,
                "n_bases_shared": 0,
                "length": 1000,
                "n": 10000,
                }
    elif project == 'ascat':
        return {"batch_size": 64,
                "cancer_types": ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],
                # "wgd": False,
                }
    elif project == "sinusoidal":
        return {"batch_size": 64,
                "classes": 3,
                "samples": 2000,
                "sig_length": 256,
                "channels": 2,
                "save_to_file": "/home/ubuntu/Documents/Notebooks/wave-LSTM_benchmark/sinusoidal.csv"
                }


if __name__ == '__main__':

    # run_simulated_example("Quiet")
    # run_ascat_example("ascat")
    run_sinusoidal_example("sinusoidal")
