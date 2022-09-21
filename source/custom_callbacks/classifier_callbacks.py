import numpy as np
from pytorch_lightning import Callback
import torch
import wandb

import random
import math

# Create custom callback for viewing sequences at the end of each validation epoch
class ValSeq(Callback):
    def __init__(self, val_samples, num_samples=16):
        super().__init__()
        self.num_samples = num_samples
        self.val_features = val_samples['feature']
        self.val_labels = val_samples['label']

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_features = self.val_features.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = torch.log_softmax(pl_module(val_features), dim=1)
        preds = torch.argmax(logits, -1)

        def custom_plot(y, title):
            print(f"y: {y.shape}")
            # data = [[idx, yi] for idx, yi in enumerate(y[0, 0, :])]

            data = [[i, random.random() + math.sin(i / 10)] for i in range(20)]
            print(f"data: {data}")
            table = wandb.Table(columns=['locus', 'count'], data=data)

            wandb.plot.line(
                table, "locus", "count", title=title
                )

        # Log the sequences, their classification, and true label
        # trainer.logger.experiment.log({
        #     "SequenceClassifications": [custom_plot(x, title=f"Pred:{pred}, Label:{y}")
        #                                 for x, pred, y in zip(val_features[:self.num_samples],
        #                                                       preds[:self.num_samples],
        #                                                       val_labels[:self.num_samples])]
        # })
        data = [[i, random.random() + math.sin(i / 10)] for i in range(100)]
        table = wandb.Table(data=data, columns=["step", "height"])
        trainer.logger.experiment.log({'line-plot1': wandb.plot.line(table, "step", "height")})

        # trainer.logger.experiment.log({
        #     "my_custom_id": custom_plot(val_features[0], title=f"Pred:{preds[0]}, Label:{val_labels[0]}")
        #         # wandb.plot.scatter(table, "locus", "count", title=f"Pred:{preds[0]}, Label:{val_labels[0]}")
        # })

