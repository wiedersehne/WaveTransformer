import torch
import numpy as np
from SignalTransformData.simulated import SinusoidalDataModule
import matplotlib.pyplot as plt




def debug_loader():

    data_module = SinusoidalDataModule(get_demo_config(),
                                       samples=2000,
                                       batch_size=128,
                                       sig_length=512)
    print(data_module)

    loader_list = {'train': data_module.train_dataloader(),
                   'test': data_module.test_dataloader(),
                   'validation': data_module.val_dataloader(),
                   }
    for key in loader_list:
        print(f'\n{key} set\n=============')
        for batch_idx, batch in enumerate(loader_list[key]):
            print(f'\nBatch {key} index {batch_idx}')
            print(f'Batch {batch.keys()}')
            print(f"Feature shape {batch['feature'].shape}, label shape {batch['label'].shape}")
            print(f"label counts {torch.unique(batch['label'], return_counts=True)}")
            print(np.unique(batch['feature'], axis=0).shape)


if __name__ == '__main__':

    debug_loader()

