import numpy as np
import TCGA
import torch
from WaveLSTM.models.attentive_autoencoder import create_sa_autoencoder


def run_ascat():

    cancer_types = ['BRCA', 'OV']  # ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG'],  # ,  #['STAD', 'COAD'],

    # Load data and filter for only the cases of interest
    dm = TCGA.data_modules.ascat.loaders.ASCATDataModule(batch_size=128, cancer_types=cancer_types,
                                                         chrom_as_channels=True)

    features, labels = [], []
    for t_batch in iter(dm.test_dataloader()):
        features.append(t_batch["feature"])
        labels.append(t_batch["label"])
    test_all = {"feature": torch.concat(features, 0),
                "label": torch.concat(labels, 0)}

    # Create modelon_validation_epoch_end
    model, trainer = create_sa_autoencoder(seq_length=dm.W, channels=dm.C,
                                           wavelet="haar",
                                           hidden_size=256, layers=1, proj_size=0, scale_embed_dim=2,
                                           r_hops=20,
                                           recursion_limit=4,
                                           num_epochs=15,
                                           validation_hook_batch=next(iter(dm.val_dataloader())),
                                           test_hook_batch=test_all,
                                           project="WaveLSTM-ascat",
                                           run_id=f"ascat-attentive-ae",
                                           )

    # Normalizing stats
    features = torch.concat([batch["feature"] for batch in dm.train_dataloader()], 0)
    mean = torch.mean(features, 0)
    std = torch.std(features, 0)
    std[std < 1e-4] = 1
    model.normalize_stats = (mean, std)

    if True:
        trainer.fit(model, datamodule=dm)
        print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    else:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.dirpath + "/CHISEL.ckpt")

    # Test
    trainer.test(model, dataloaders=dm.test_dataloader())

    # Dump attentie auto-encoder's state to file so it can be used as a pre-trained model for survival example
    torch.save(model.a_encoder.state_dict(), "./ascat_sae_pretrained.pt")



if __name__ == '__main__':

    run_ascat()
