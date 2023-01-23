import torch
import TCGA

from source.vec2seq import create_vec2seq
from source.model.decoder.rnn_wavelet import WaveletLSTM
from source.model.encoder.sequence_encoder import SequenceEncoder

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make .size method in DataModules so model code can be streamlined

# TODO: make a proper configuration (.yaml or whatever)
project = 'Quiet'
if project == 'Quiet':
    sim_config = {"steps": 2,
                  "classes": 10,
                  "n_class_bases": 4,
                  "n_bases_shared": 0,
                  "length": 60,
                  "n": 10000,
                  }
elif project == 'Unstable':
    sim_config = {"steps": 2,
                  "classes": 10,
                  "n_class_bases": 10,
                  "n_bases_shared": 0,
                  "length": 20,
                  "n": 10000,
                  }


def experiment(simulated=False):

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.simulated.MarkovDataModule(**sim_config)
    # print(dm)

    latent_dimension = sim_config["classes"]
    seq_length = sim_config["length"]
    chromosomes = 1
    strands = 2

    # Create encoder
    encoder = SequenceEncoder(in_features=seq_length, out_features=latent_dimension, n_hidden=128, n_layers=3,
                              dropout=0.6, bidirectional=True, in_channels=2, out_channels=2,
                              kernel_size=3, stride=5, padding=1)

    # Create decoder
    embedding = "AvgPool1d"
    decoder = WaveletLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                          hid_dim=512, layers=1, bidirectional=True, embedding=embedding)

    # Validation and test batch (TODO: update to all set)
    val_iterator = iter(dm.val_dataloader())
    test_iterator = iter(dm.val_dataloader())

    # Create model
    model, trainer = create_vec2seq(encoder_model=encoder, decoder_model=decoder,
                                    wavelet='haar',  # 'bior4.4',  # 'coif4'
                                    coarse_skip=0, recursion_limit=None,
                                    auto_reccurent=True, teacher_forcing_ratio=0.5,
                                    run_id=project + f"_autoreccurent_hdim{decoder.h_dim}_{embedding}_haar",
                                    num_epochs=20,
                                    validation_hook_batch=next(val_iterator),
                                    test_hook_batch=next(test_iterator)
                                    )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    experiment(simulated=True)
