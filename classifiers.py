import torch
import CNN.cnn
from CNN.cnn import CNNKyticko
import NEURAL_PCA
from NEURAL_PCA.neural_pca import NeuralPCA, PCA_dataset
import logging
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(levelname)s): %(funcName)s:%(lineno)d %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
#logger.disabled = True


class Classifier:
    def __init__(self, hparams):
        self.hparams = hparams

    def predict(self, filename):
        raise NotImplemented("Implement me")

    def train(self, hparams):
        raise NotImplemented("Implement me")

    def save(self, save_path):
        raise NotImplemented("Implement me")


# todo: prepare classes
class NeuralPCAClassifier(Classifier):
    def __init__(self, train, hparams):
        super(NeuralPCAClassifier).__init__(hparams)
        self.data = PCA_dataset(root_dir=hparams["root_dir"], batch_size=16)
        U, mean = self.data.get_U_mean()
        if self.train:
            pass
        else:
            pass

    def predict(self):
        # todo: transform an image
        # todo: normalize (as pca does) the image
        pass


class MAPClassifier(Classifier):
    def __init__(self, hparams):
        super(MAPClassifier).__init__(hparams)
        # train model and store it
        if self.train:
            ## Training model
            #
            #

            ## Storing the model
            #
            #
            pass
        # load and test model
        else:
            pass

    def predict(self, filename):
        pass

    def train(self, hparams):
        pass

    def save(self, save_path):
        pass


class CNNClassifier(Classifier):
    def __init__(self, hparams):
        super(CNNClassifier).__init__(hparams)
        if self.hparams["train"]:
            # train and store the model
            CNN.cnn.main(self.hparams)

        # load and test model
        if self.hparams["eval"]:
            path = os.path.join(self.hparams['model_dir'] + '/', self.hparams['model_name'])
            logger.debug(f"Loading model from {path}...")
            self.model = CNNKyticko()
            # load a model (in models/<name>)
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logger.debug("Model has been loaded successfully")

    def predict(self, filename):
        image = Image.open(filename)
        # load an image & transform to a tensor + convert to grayscale
        transform = transforms.Compose([
                np.ascontiguousarray,
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
        ])
        image = np.array(image)
        image = transform(image)
        soft = self.model(image)

        # soft, hard decision
        return soft.ravel().numpy()[0], int(bool(soft.ravel().numpy()[0] > .5))

