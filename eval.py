from glob import glob
import yaml
from classifiers import NeuralPCAClassifier, MAPClassifier, CNNClassifier
import pytorch_lightning as pl
pl.utilities.seed.seed_everything(42)


def main(hparams):
    dicts = {"Neural_PCA": [NeuralPCAClassifier, '.png'], "MAP": [MAPClassifier, '.wav'], "CNN": [CNNClassifier, 'png']}

    for key, classifier in dicts.items():
        classifier, file_format = classifier
        if hparams[key]["train"]:
            interface = classifier(hparams=hparams[key])

        if hparams[key]["eval"]:
            interface = classifier(hparams=hparams[key])
            with open(f"{key}_predictions.txt", 'w') as f:
                for file in glob(f"{hparams['eval_dataset']}/*{file_format}"):
                    soft, hard = interface.predict(file)
                    print(f"{file} {soft:.2f} {hard}", file=f)


# parse hyperpyyaml
with open("hp.yaml", 'r') as stream:
    hparams = yaml.safe_load(stream)

#print("ZDES':", hparams)
""" hparams = {
    "CNN": {"train" : False,
            "eval" : False},
    "MAP": {"train": False, # if not true -- load
            "eval": True,
            "model_dir": "./models",
            "dataset_dir": {"non_target": "./dataset/non_target_train", "target": "./dataset/target_train"},
            "dev_dataset" : {"non_target": "./dataset/non_target_dev", "target": "./dataset/target_dev"},

            "model_name": {"target": "bgmm_target.pkl", "non_target": "bgmm_non_target.pkl"},
            "eval_dir": "./eval",
            "eval_out": "output",
            "GPU": 0,
            "root_dir": ".",
            "wandb_entity": "skuratovich"},
    "Neural_PCA" : {"train" : True,
                    "eval" : False}
} """

main(hparams)
