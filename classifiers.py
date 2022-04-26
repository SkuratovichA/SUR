
class Classifier:
    def __init__(self, train, hparams):
        self.train = train
        self.hparams = hparams

    def predict(self, filename, hparams):
        raise NotImplemented("Implement me")

    def train(self, hparams):
        raise NotImplemented("Implement me")

    def save(self, save_path):
        raise NotImplemented("Implement me")


# todo: prepare classes
class NeuralPCAClassifier(Classifier):
    def __init__(self, train, hparams):
        super(NeuralPCAClassifier).__init__(train, hparams)

    def predict(self, filename, hparams):
        pass

    def train(self, hparams):
        pass

    def save(self, save_path):
        pass


class MAPClassifier(Classifier):
    def __init__(self, train, hparams):
        super(MAPClassifier).__init__(train, hparams)

    def predict(self, filename, hparams):
        pass

    def train(self, hparams):
        pass

    def save(self, save_path):
        pass


class CNNClassifier(Classifier):
    def __init__(self, train, hparams):
        super(CNNClassifier).__init__(train, hparams)

    def predict(self, filename, hparams):
        pass

    def train(self, hparams):
        pass

    def save(self, save_path):
        pass