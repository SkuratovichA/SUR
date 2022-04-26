from glob import glob
import yaml
from classifiers import NeuralPCAClassifier, MAPClassifier, CNNClassifier
import pytorch_lightning as pl
pl.utilities.seed.seed_everything(42)
import speechbrain.data_io.data_io
from speechbrain.hyperpyyaml.core import load_hyperpyyaml

def main(hparams):
    dicts = {"Neural_PCA": NeuralPCAClassifier, "MAP": MAPClassifier, "CNN": CNNClassifier}

    for key, classifier in dicts:

        if hparams[key]["train"]:
            interface = classifier(train=True, hparams=hparams[key])
            interface.train()
            interface.save()

        if hparams[key]["eval"]:
            interface = classifier(train=False, hparams=hparams[key])
            for file in glob(hparams[key]["eval_dataset"]):
                soft, hard = interface.predict(file)
                print(f"{file}  {soft:.2f}  {hard}", file=file)


hparams = yaml.safe_load("hp.yml")
main(hparams)