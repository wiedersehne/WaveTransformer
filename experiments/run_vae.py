import numpy as np
import torch
from source.vae import create_vanilla_vae
# Decoder
from source.models.linear_decoder import WrappedLinearDecoder
from source.models.sequence_encoder import SequenceEncoder
from source.helpers.edges2wavelet import edges2wavelet

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make .size method in DataModules

def simulated_dm():
    from data.simulatedMarkov.loader import MarkovDataModule as DataModule
    return DataModule(classes=3, steps=2, n=10000, length=20, n_class_bases=2, n_bases_shared=0)


def ascat_dm():
    from data.ASCAT.loader import ASCATDataModule as DataModule
    cancer_types = ['OV', 'STAD']   # ['STAD', 'COAD']
    return DataModule(cancer_types=cancer_types, batch_size=32, custom_edges=edges2wavelet)


def experiment(simulated=False):

    # Load data and filter for only the cases of interest
    if simulated:
        raise NotImplementedError
        dm = simulated_dm()
        stacked_bases = np.vstack(dm.bases)
    else:
        dm = ascat_dm()

    print(dm)
    latent_dimension = 40 if not simulated else stacked_bases.shape[0]
    seq_length = 11642 if not simulated else 20

    # Create encoder
    encoder = SequenceEncoder(in_features=seq_length, out_features=latent_dimension, n_hidden=128, n_layers=3,
                              dropout=0.6, bidirectional=True, in_channels=2, out_channels=2,
                              kernel_size=3, stride=5, padding=1)

    # Create decoder
    decoder = WrappedLinearDecoder(in_features=latent_dimension, out_features=seq_length, strands=2,
                                   hidden_features=32)

    # Validation set
    val_iterator = iter(dm.val_dataloader())
    # batches = 3
    # for i in range(batches):
    #     feature, label = next(val_iterator)

    # Create model
    model, trainer = create_vanilla_vae(encoder_model=encoder, decoder_model=decoder,
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
