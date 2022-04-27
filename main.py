from glob import glob
import yaml
from classifiers import NeuralPCAClassifier, MAPClassifier, CNNClassifier
import pytorch_lightning as pl
pl.utilities.seed.seed_everything(42)


def main(hparams):
    dicts = {"Neural_PCA": [NeuralPCAClassifier, '.png'], "MAP": [MAPClassifier, '.wav'], "CNN": [CNNClassifier, 'png']}

    for key, classifier in dicts.items():
        classifier, file_format = classifier
        interface = classifier(hparams=hparams[key])

        if hparams[key]["eval"]:
            with open(f"{key}_predictions.txt", 'w') as f:
                for file in glob(f"{hparams['eval_dataset']}/*{file_format}"):
                    soft, hard = interface.predict(file)
                    print(f"{file} {soft:.2f} {hard}", file=f)


# parse hyperpyyaml
with open("hyperparams.yaml", 'r') as stream:
    hparams = yaml.safe_load(stream)
main(hparams)
