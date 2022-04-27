from glob import glob
import yaml
from classifiers import NeuralPCAClassifier, MAPClassifier, CNNClassifier
import pytorch_lightning as pl
pl.utilities.seed.seed_everything(42)


def main(hparams):
    dicts = {"Neural_PCA": NeuralPCAClassifier, "MAP": MAPClassifier, "CNN": CNNClassifier}

    for key, classifier in dicts:

        if hparams[key]["train"]:
            interface = classifier(train=True, hparams=hparams[key])

        if hparams[key]["eval"]:
            interface = classifier(train=False, hparams=hparams[key])
            with open(f"{key}_predictions.txt", 'w') as f:
                for file in glob(hparams["eval_dataset"]):
                    soft, hard = interface.predict(file)
                    print(f"{file} {soft:.2f} {hard}", file=f)


# TODO: parse hyperpyyaml here
hparams = yaml.safe_load('hp.yaml')
hparams = {
    "CNN": {"train" : False,
            "eval" : False},
    "MAP": {"train": True,
            "eval" : True},
    "Neural_PCA" : {"train" : False,
                    "eval" : False}
}

main(hparams)
