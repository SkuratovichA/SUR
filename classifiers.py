import torch
import CNN.cnn
from CNN.cnn import CNNKyticko
import MAP.map
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


class NeuralPCAClassifier(Classifier):
    def __init__(self, train, hparams):
        super(NeuralPCAClassifier).__init__(train, hparams)
        self.data = PCA_dataset(root_dir=hparams["root_dir"], batch_size=16)
        U, mean = self.data.get_U_mean()
        if self.hparams["train"]:
            pass

        if self.hparams["eval"]:
            pass

    def predict(self, filename):
        # todo: transform an image
        # todo: normalize (as pca does) the image
        pass


class MAPClassifier(Classifier):
    def __init__(self, hparams):
        super(MAPClassifier).__init__(hparams)
        self.model = MAP.map.main(self.hparams)
        # train model and store it
        if self.hparams["train"]:
            self.model.train()
            self.model.save()

        # load and test model
        if self.hparams["eval"]:
            self.model.load()

    def predict(self, filename):
        soft, hard = self.model.evaluate(filename)


class CNNClassifier(Classifier):
    def __init__(self, hparams):
        super(CNNClassifier).__init__(hparams)

        # train and save the model
        if self.hparams["train"]:
            CNN.cnn.main(self.hparams)

        # load the trained model
        if self.hparams["eval"]:
            path = os.path.join(self.hparams['model_dir'], self.hparams['model_name'])
            logger.debug(f"Loading model from {path}...")
            self.model = CNNKyticko()
            # load a model (in models/<name>)
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.model.eval()
            logger.debug("Model has been loaded successfully")

    def predict(self, filename):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        image = transform(np.array(Image.open(filename))).unsqueeze(0)
        soft = self.model(image).detach().ravel()[0]

        # soft, hard decision
        return soft.ravel().numpy()[0], int(soft > .5)