import os
import torch
import CNN.cnn
import MAP.map
import NEURAL_PCA
import numpy as np
from PIL import Image
from CNN.cnn import CNNKyticko
import torchvision.transforms as transforms
from NEURAL_PCA.neural_pca import NeuralPCA, PCADataset

import logging

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
        super(NeuralPCAClassifier).__init__(hparams)

        if self.hparams["train"]:
            NEURAL_PCA.neural_pca.main(self.hparams)

        if self.hparams["eval"]:
            self.u, self.mean = PCADataset(root_dir=hparams["dataset_dir"], batch_size=16).get_u_mean()
            path = os.path.join(hparams["model_dir"], hparams["model_name"])
            self.model = NeuralPCA()
            logger.debug(f"model path: {path}")
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.model.eval()

    def predict(self, filename):
        transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Grayscale(num_output_channels=1)
        ])
        image = (NEURAL_PCA.neural_pca.convert_pil_to_tensor_and_transform(filename, transform))  # numpy form
        image = torch.tensor((image - self.mean.numpy()).dot(self.u.T.numpy())).unsqueeze(0)

        soft = self.model(image).detach().ravel()[0]
        return soft, int(soft > .5)


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
