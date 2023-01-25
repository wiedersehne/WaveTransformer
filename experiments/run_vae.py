import numpy as np
import torch
import TCGA

from source.vae import create_vae
# Decoder
from source.model.decoder.linear import LinearDecoder
from source.model.encoder.sequence_encoder import SequenceEncoder

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make .size method in DataModules


def experiment(simulated=False):

    # Load data and filter for only the cases of interest
    if simulated:
        n_classes = 4
        dm = TCGA.data_modules.simulated.MarkovDataModule(2, classes=n_classes, n=10000, length=20, n_class_bases=2,
                                                          n_bases_shared=0)
        # stacked_bases = np.vstack(dm.bases)
    else:
        cancer_types = ['OV', 'STAD']  # ['STAD', 'COAD']
        n_classes = len(cancer_types)
        dm = TCGA.data_modules.ascat.ASCATDataModule(cancer_types=cancer_types,
                                                     batch_size=128)  # , custom_edges=edges2wavelet)

    print(dm)
    latent_dimension = 30 if not simulated else 2
    seq_length = 1000 if not simulated else 20
    chromosomes = 23 if not simulated else 1

    # Create encoder
    encoder = SequenceEncoder(in_features=seq_length, out_features=latent_dimension, n_hidden=128, n_layers=3,
                              dropout=0.6, bidirectional=True, in_channels=2, out_channels=2,
                              kernel_size=3, stride=5, padding=1)

    # Create decoder
    decoder = LinearDecoder(in_features=latent_dimension,
                            out_features=seq_length,
                            strands=2,
                            chromosomes=chromosomes,
                            hidden_features=32)

    # Validation set
    val_iterator = iter(dm.val_dataloader())

    # Create model
    model, trainer = create_vae(encoder_model=encoder, decoder_model=decoder,
                                latent_dim=latent_dimension, kld_weight=0.,
                                validation_hook_batch=next(val_iterator),
                                )

    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    experiment(simulated=False)
