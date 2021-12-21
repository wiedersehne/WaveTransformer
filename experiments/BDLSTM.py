import pandas as pd
import torch
from data.pipeline import cn_pipeline_constructor
from models.CNNLSTM import model_constructor
from sklearn.metrics import (classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # Load data and filter for only the cases of interest
    data_module = cn_pipeline_constructor()

    # Create model
    model, trainer = model_constructor(n_classes=data_module.n_classes)

    # Train model
    trainer.fit(model, datamodule=data_module)
    # trainer.test()                                        # TODO: why doesn't this work?
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
    trained_model.freeze()

    return trained_model, data_module


def test(trained_model, data_module):

    def show_confusion_matrix(conf_m):
        hmap = sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True cancer type')
        plt.xlabel('Predicted cancer type')
        plt.show()

    predictions, labels = [], []
    for idx_batch, batch in enumerate(data_module.test_dataloader()):
        # print(f'batch {idx_batch}')
        x_test = trained_model.batch_to_data(batch)[0].to(device)
        _, output = trained_model(x_test)
        prediction = torch.argmax(output, dim=1).tolist()

        label = batch['cancer_type'].tolist()
        for i in range(len(label)):
            predictions.append(prediction[i])
            labels.append((label[i]))

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
        def forward(self, x):
            return x

    batch = next(iter(data_module.train_dataloader()))
    print(batch)

    # Remove prediction head
    model_ = copy.deepcopy(model)
    model_.model.fc = Identity()

    meta = model_.batch_to_data(batch)[1].detach().cpu().numpy()        # Get labels for plotting
    x_test = model_.batch_to_data(batch)[0].to(device)

    model_.write_embeddings(x=x_test.to(device))
    model_.create_tensorboard_log(metadata=meta)

    return


if __name__ == '__main__':
    # TODO: Clean up pipeline more,
    # TODO: Change pipeline to just output model_in, model_out = @property(?)_wrapper(dictionary)
    # TODO: Then make new pipeline for simulated examples
    # TODO: Simulated example. Two clusters with 5 transitions each. Visualise latent separation

    trained_model, data_module = train()

    test(trained_model, data_module)

    visualise_latent(trained_model, data_module)
