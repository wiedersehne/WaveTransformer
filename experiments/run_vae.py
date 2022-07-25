import pandas as pd
import torch
from source.vae import model_constructor
import copy

# Set device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vae(data_module, h, seq_len, kernels=None):

    # Construct model
    model, trainer = model_constructor(latent_dim=h,
                                       seq_length=seq_len,
                                       kernels=kernels
                                       )

    # Train model
    trainer.fit(model, datamodule=data_module)

    # trainer.test()                                        # TODO: why doesn't this work?
    trained_model = model.load_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
    trained_model.freeze()

    return trained_model


def test(trained_model, data_module):

    if False:
        # TODO: replace with plotting
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

        cm = confusion_matrix(labels, predictions)
        df_cm = pd.DataFrame(
            cm, index=data_module.label_encoder.classes_, columns=data_module.label_encoder.classes_
        )
        show_confusion_matrix(df_cm)


def visualise_latent(model, data_module):

    batch_train = next(iter(data_module.train_dataloader()))
    batch_test = next(iter(data_module.test_dataloader()))
    batch_val = next(iter(data_module.val_dataloader()))

    sequences = torch.cat((batch_train['feature'],
                           batch_test['feature'],
                           batch_val['feature']
                           ), 0)
    labels = torch.cat((batch_train['label'],
                        batch_test['label'],
                        batch_val['label']
                        ), 0)
    #print(sequences.shape)

    model = copy.deepcopy(model)
    model.write_embeddings(x=sequences.to(device), y=labels)
    model.create_tensorboard_log(metadata=labels.detach().cpu().numpy())


def main(experiment='simulated'):

    if experiment == 'simulated':
        from data.pipeline_simulatedASCAT import DataModuleMDP as DataModule
    elif experiment == 'real':
        from data.pipeline_ascat import DataModuleASCAT as DataModule
    else:
        raise NotImplementedError

    # Load data and filter for only the cases of interest
    data_module = DataModule()
    if data_module.simulated() is True:
        h, kernels = data_module.n_bases, data_module.bases
    else:
        h, kernels = 50, None

    #
    trained_model = train_vae(data_module, h, data_module.seq_length, kernels)

    #
    # test(trained_model, data_module)

    #
    visualise_latent(trained_model, data_module)


if __name__ == '__main__':
    # TODO: Remove relative imports and add package imports

    main(experiment='real')
