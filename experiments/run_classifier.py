import torch
from source.classifier import create_classifier
from source.helpers.edges2wavelet import edges2wavelet
from source.models import SequenceEncoder

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def test(trained_model, data_module):
#     # TODO: replace with test hook
#
#     def show_confusion_matrix(conf_m):
#         hmap = sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues")
#         hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#         hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
#         plt.ylabel('True label')
#         plt.xlabel('Predicted label')
#         plt.show()
#
#     predictions, labels = [], []
#     for idx_batch, batch in enumerate(data_module.test_dataloader()):
#         x_test = batch['feature'].to(device)
#         _, output = trained_model(x_test)
#         prediction = torch.argmax(output, dim=1).tolist()
#
#         label = batch['label'].tolist()
#         for i in range(len(label)):
#             predictions.append(prediction[i])
#             labels.append((label[i]))
#
#     print(data_module.label_encoder.classes_)
#     print(type(data_module.label_encoder.classes_[0]))
#
#     print(
#         classification_report(labels, predictions, target_names=data_module.label_encoder.classes_)
#     )
#
#     cm = confusion_matrix(labels, predictions)
#     df_cm = pd.DataFrame(
#         cm, index=data_module.label_encoder.classes_, columns=data_module.label_encoder.classes_
#     )
#     show_confusion_matrix(df_cm)
#
#
# def visualise_latent(model, data_module):
#     # TODO: replace with validation hook
#
#     class Identity(torch.nn.Module):
#         def __init__(self):
#             super(Identity, self).__init__()
#
#         @staticmethod
#         def forward(x):
#             return x
#
#     # TODO: plot _all_ data, train+test+val
#     batch = next(iter(data_module.test_dataloader()))
#     sequences = batch['feature']
#     labels = batch['label']
#
#     # No head
#     # Remove prediction head
#     model_nohead = copy.deepcopy(model)
#     model_nohead.model.fc = Identity()
#     model_nohead.write_embeddings(x=sequences.to(device), y=labels)
#     model_nohead.create_tensorboard_log(metadata=labels.detach().cpu().numpy())
#
#     return


def simulated_dm():
    from data.simulatedMarkov.loader import MarkovDataModule as DataModule
    return DataModule(classes=3, steps=4, n=2000, length=20, n_class_bases=2, n_bases_shared=0)


def ascat_dm():
    from data.ASCAT.loader import ASCATDataModule as DataModule
    cancer_types = ['STAD', 'COAD']  # ['OV', 'STAD']
    return DataModule(cancer_types=cancer_types, custom_edges=edges2wavelet)


def main(simulated=False):

    # Load data and filter for only the cases of interest
    if simulated:
        raise NotImplementedError
        dm = simulated_dm()
        chrom_channels, out_channels = 1, 1
        seq_length = 20

    else:
        dm = ascat_dm()
        chrom_channels, out_channels = 2, 2
        seq_length = 11642

    print(dm)

    encoder = SequenceEncoder(in_features=seq_length, out_features=dm.num_cancer_types, n_hidden=128, n_layers=3,
                              dropout=0.5, bidirectional=True, in_channels=2, out_channels=2,
                              kernel_size=3, stride=5, padding=1)

    # Create model
    model, trainer = create_classifier(model=encoder,
                                       # validation_hook_batch=next(iter(dm.val_dataloader())),
                                       )
    # Train model
    trainer.fit(model, datamodule=dm)
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
    trained_model.freeze()

    # Test model
    trainer.test(trained_model, dataloaders=dm.test_dataloader())


if __name__ == '__main__':

    main(simulated=False)

