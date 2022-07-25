import pandas as pd
import torch
from source.classifier import model_constructor
from sklearn.metrics import (classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from experiments.configs.config import extern

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data_module):
    # Create model
    model, trainer = model_constructor(n_classes=data_module.n_classes, seq_length=data_module.seq_length)

    # Train model
    trainer.fit(model, datamodule=data_module)
    # trainer.test()                                        # TODO: why doesn't this work?
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
    trained_model.freeze()

    return trained_model


def test(trained_model, data_module):

    def show_confusion_matrix(conf_m):
        hmap = sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    predictions, labels = [], []
    for idx_batch, batch in enumerate(data_module.test_dataloader()):
        x_test = batch['feature'].to(device)
        _, output = trained_model(x_test)
        prediction = torch.argmax(output, dim=1).tolist()

        label = batch['label'].tolist()
        for i in range(len(label)):
            predictions.append(prediction[i])
            labels.append((label[i]))

    print(data_module.label_encoder.classes_)
    print(type(data_module.label_encoder.classes_[0]))

    print(
        classification_report(labels, predictions, target_names=data_module.label_encoder.classes_)
    )

    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(
        cm, index=data_module.label_encoder.classes_, columns=data_module.label_encoder.classes_
    )
    show_confusion_matrix(df_cm)


def visualise_latent(model, data_module):

    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        @staticmethod
        def forward(x):
            return x

    # TODO: plot _all_ data, train+test+val
    batch = next(iter(data_module.test_dataloader()))
    sequences = batch['feature']
    labels = batch['label']

    # No head
    # Remove prediction head
    model_nohead = copy.deepcopy(model)
    model_nohead.model.fc = Identity()
    model_nohead.write_embeddings(x=sequences.to(device), y=labels)
    model_nohead.create_tensorboard_log(metadata=labels.detach().cpu().numpy())

    return


@extern
def main(experiment):
    if experiment == 'simulated':
        from data.pipeline_simulatedASCAT import DataModuleMDP as DataModule
    elif experiment == 'ascat':
        from data.pipeline_ascat import DataModuleASCAT as DataModule
    else:
        raise NotImplementedError

    # Load data and filter for only the cases of interest
    data_module = DataModule()

    #
    trained_model = train(data_module)

    #
    test(trained_model, data_module)

    #
    visualise_latent(trained_model, data_module)


if __name__ == '__main__':
    # TODO: Keep cleaning,
    # TODO: Build unsupervised model
    # TODO: Remove relative imports and add package imports

    main(experiment='ascat')
