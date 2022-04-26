from glob import glob
from classifiers import NeuralPCAClassifier, MAPClassifier, CNNClassifier



def main(hparams):
    dicts = {"NeuralPCA": NeuralPCAClassifier, "MAP": MAPClassifier, "CNN": CNNClassifier}

    for key, classifier in dicts:

        if hparams[key]["train"]:
            interface = classifier(train=True, hparams=hparams[key])
            interface.train(hparams)
            interface.save(hparams[key][f"{key}_path"])

        if hparams[key]["eval"]:
            interface = classifier(train=False, hparams=hparams[key])
            for file in glob(hparams["eval_dataset"]):
                soft, hard = interface.predict(file, hparams)
                print(f"{file}  {soft:.2f}  {hard}", file=file)


# TODO: parse hyperpyyaml here
hparams = {}
main(hparams)