from glob import glob
import yaml
from classifiers import NeuralPCAClassifier, MAPClassifier, CNNClassifier
import pytorch_lightning as pl
import sys

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(levelname)s): %(funcName)s:%(lineno)d %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
#logger.disabled = True

pl.utilities.seed.seed_everything(42)

def main(hparams):
    dicts = {"Neural_PCA": [NeuralPCAClassifier, '.png'], "MAP": [MAPClassifier, '.wav'], "CNN": [CNNClassifier, 'png']}

    for key, classifier in dicts.items():
        classifier, file_format = classifier
        interface = classifier(hparams=hparams[key])
        logger.debug(f"classifier: {key}")

        if hparams[key]["eval"]:
            logger.debug("evaluating...")
            with open(f"{key}_predictions.txt", 'w') as f:
                for file in glob(f"{hparams['eval_dir']}/*{file_format}"):
                    soft, hard = interface.predict(file)
                    print(f"{file} {soft:.2f} {hard}", file=f)
            logger.debug(" DONE")


# parse hyperpyyaml
with open(sys.argv[1], 'r') as stream:
    hparams = yaml.safe_load(stream)
hparams |= hparams["default"]
del hparams['default']
# print(hparams.keys())
# exit(0)
main(hparams)
