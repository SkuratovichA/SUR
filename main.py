import os
import seaborn as sns
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import librosa as lb
from librosa.feature import mfcc


class Dataset:
    def __init__(self, dataset_type):
        """
        :param dataset_type: one of ['train', 'eval']. For 'train' filename of type `f201_s1_i2_smth.wav` is considered.
                             For evaluation, filename is `smth.wav`.
        """
        self.samples = {}
        self.dirfilter = lambda x: os.path.splitext(os.path.basename(x))  # 'smth/honza.wav' -> ['honza', 'wav']
        self.file_extension = lambda x: self.dirfilter(x)[1][1:]
        self.session = lambda x: x
        match dataset_type:
            case 'train':
                self.sample_name = lambda x: self.dirfilter(x)[0].split("_")[0]  # 'smth/smth/f201_s1_i2_smth.wav' -> 'f201' # noqa
                self.session = lambda x: self.dirfilter(x)[0].split("_")[1]  # 'smth/smth/f201_s1_i2_smth.wav' -> 's1' # noqa
            case 'eval':
                self.sample_name = lambda x: self.dirfilter(x)[0]  # just an identity: ['honza', 'wav'] -> 'honza'
            case _ as val:
                raise ValueError(f"{val} is not supported in the constructor. Expecting one of 'train', 'eval'")

    def get_dataset(self, directories, extensions=None):
        r"""
        Creates a dataset files in specified directories.

        :param directories: list of directories or a string representing a directory
        :param extensions: one of 'wav', 'png'. If None, both are included.
        :return: dictionary of type {'<person id>' : {'png' : filename.png, 'wav' : filename.wav}}
        """
        extensions = [extensions] if isinstance(extensions, str) else extensions
        fnames = lambda x: x
        match extensions:
            case ['wav']:
                fnames = lambda d: glob(d + '/*.wav')
            case ['png']:
                fnames = lambda d: glob(d + "/*.png")
            case ['wav', 'png'] | None:
                fnames = lambda d: glob(d + "/*.png") + glob(d + '/*.wav')
            case _ as err:
                raise ValueError(f"{err} is not supported. Expecting one of ['wav', 'png', ['wav', 'png']]")
        samples = {}
        directories = [directories] if isinstance(directories, str) else directories
        for directory in directories:
            for f in fnames(directory):
                sample_name = self.sample_name(f)
                session = self.session(f)
                match self.file_extension(f):
                    case 'wav':
                        sig, rate = lb.load(f, sr=16000)
                        assert rate == 16000, f"sample rate must be 16kHz, got {rate} for {f}"
                        sig = sig[26000:]
                        sig = (sig - sig.mean()) / np.abs(sig).max()
                        sig = mfcc(y=sig, sr=rate)
                        dato = {self.file_extension(f): sig}
                    case 'png':
                        # TODO: add image extraction
                        dato = {self.file_extension(f): f}
                if sample_name not in samples:
                    samples[sample_name] = {}
                if session not in samples[sample_name]:
                    samples[sample_name][session] = dato # noqa
                else:
                    samples[sample_name][session].update(dato)
        return samples

    @staticmethod
    def plot_all_signals(dataset, number_of_mfcc_features=1):
        r"""
        Plots mfcc coefficients using seaborn. May take a while.

        :param dataset: dataset produced by `get_dataset()`.
        :param number_of_mfcc_features: number of mfcc features to plot.
        :return:
        """
        data = []
        for person in dataset.items():
            for session in person[1].items():
                for i in range(number_of_mfcc_features):
                    data.append(session[1]['wav'][i])
        ax = sns.lineplot(data=data, alpha=1, linewidth=0.05, legend=False)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        plt.show()


def main():
    datasetter = Dataset('train')
    dataset = datasetter.get_dataset(['non_target_dev', 'non_target_train'])
    datasetter.plot_all_signals(dataset)


if __name__ == "__main__":
    main()
