import torch
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter


class EmbeddingModule(torch.nn.Module):
    """
    This class extracts, reads, and writes data embeddings using the model this class is inherited from. Meant to work with
        Tensorboard's Embedding Viewer (https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).
        When using with a 3 channel image input and a pretrained model from torchvision.models please use the
        following pre-processing pipeline:

        transforms.Compose([transforms.Resize(imsize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]) ## As per torchvision docs

        Args:
            model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
            seq_folder (str): The folder path where the input data elements should be written to
            embs_folder (str): The folder path where the output embeddings should be written to
            tensorboard_folder (str): The folder path where the resulting Tensorboard log should be written to
            experiment_name (str): The name of the experiment to use as the log name
    """

    def __init__(self,
                 seq_folder='experiments/logs/embeddings/imgs',
                 embs_folder='experiments/logs/embeddings/embs',
                 tensorboard_folder='experiments/logs/embeddings/tb',
                 experiment_name=None):

        super(EmbeddingModule, self).__init__()

        # self.model = model
        # self.model.eval()
        self.model = None

        self.seq_folder = seq_folder
        self.embs_folder = embs_folder
        self.tensorboard_folder = tensorboard_folder

        self.name = experiment_name

        self.writer = None

    def generate_embeddings(self, x):
        '''
        Generate embeddings for an input batched tensor. Ensure head layer is removed.

        Args:
            x (torch.Tensor) : A batched pytorch tensor

        Returns:
            (torch.Tensor): The output of self.model against x
        '''
        embedding = self.model(x)
        return (embedding)

    def write_embeddings(self, x, outsize=(28, 28)):
        '''
        Generate embeddings for an input batched tensor and write inputs and
        embeddings to self.seq_folder and self.embs_folder respectively.

        Inputs and outputs will be stored in .npy format with randomly generated
        matching filenames for retrieval

        Args:
            x (torch.Tensor) : An input batched tensor that can be consumed by self.model
            outsize (tuple(int, int)) : A tuple indicating the size that input data arrays should be
            written out to

        Returns:
            (bool) : True if writing was succesful

        '''

        #assert len(os.listdir(self.seq_folder)) == 0, "Images folder must be empty"
        #assert len(os.listdir(self.embs_folder)) == 0, "Embeddings folder must be empty"

        #print(f'input {x.detach().cpu().numpy().shape}')

        # Generate embeddings
        embs = self.generate_embeddings(x)

        # Detach from graph
        embs = embs.detach().cpu().numpy()
        #print(f'embeddings {len(embs)}, {embs[0]}')

        # Start writing to output folders
        for i in range(len(embs)):
            key = str(np.random.random())[-7:]
            np.save(self.seq_folder + r"/" + key + '.npy', tensor2np(x[i], outsize))
            np.save(self.embs_folder + r"/" + key + '.npy', embs[i])
        return (True)

    def _create_writer(self, name):
        '''
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer

        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.name

        Returns:
            (bool): True if writer was created succesfully

        '''

        if self.name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.name

        dir_name = os.path.join(self.tensorboard_folder,
                                name)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))

        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)
        return (True)

    def create_tensorboard_log(self, metadata=None):

        '''
        Write all images and embeddings from seq_folder and embs_folder into a tensorboard log
        '''

        if self.writer is None:
            self._create_writer(self.name)

        ## Read in
        all_embeddings = [np.load(os.path.join(self.embs_folder, p)) for p in os.listdir(self.embs_folder) if
                          p.endswith('.npy')]
        all_images = [np.load(os.path.join(self.seq_folder, p)) for p in os.listdir(self.seq_folder) if
                      p.endswith('.npy')]
        all_images = [np.moveaxis(a, 2, 0) for a in all_images]  # (HWC) -> (CHW)

        ## Delete files

        ## Stack into tensors
        all_embeddings = torch.Tensor(all_embeddings)
        all_images = torch.Tensor(all_images)

        print(all_embeddings.shape)
        print(all_images.shape)

        self.writer.add_embedding(all_embeddings, metadata=metadata)#, label_img=all_images)


def tensor2np(tensor, resize_to=None):
    '''
    Convert an image tensor to a numpy image array and resize

    Args:
        tensor (torch.Tensor): The input tensor that should be converted
        resize_to (tuple(int, int)): The desired output size of the array

    Returns:
        (np.ndarray): The input tensor converted to a channel last resized array
    '''
    out_array = tensor.detach().cpu().numpy()
    return out_array
    out_array = np.moveaxis(out_array, 0, 2)  # (CHW) -> (HWC)

    if resize_to is not None:
        out_array = cv2.resize(out_array, dsize=resize_to, interpolation=cv2.INTER_CUBIC)

    return (out_array)