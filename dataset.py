import os
from glob import glob
import numpy as np
import librosa as lb
from librosa.feature import mfcc


class Dataset:
    def __init__(self, directories, extensions=None):
        r"""
        Creates a dataset files in specified directories.

        :param directories: list of directories or a string representing a directory
        :param extensions: one of 'wav', 'png'. If None, both are included.
        :return: dictionary of type {'<person id>' : {'png' : filename.png, 'wav' : filename.wav}}
        """
        self.wavs = {}
        self.pngs = {}
        self.samples = {}
        self.dirfilter = lambda x: os.path.splitext(os.path.basename(x))  # 'smth/honza.wav' -> ['honza', 'wav']
        self.file_extension = lambda x: self.dirfilter(x)[1][1:]

        extensions = [extensions] if isinstance(extensions, str) else extensions
        match extensions:
            case ['wav']:
                fnames = lambda d: glob(d + '/*.wav')  # noqa
            case ['png']:
                fnames = lambda d: glob(d + "/*.png")  # noqa
            case ['wav', 'png'] | None:
                fnames = lambda d: glob(d + "/*.png") + glob(d + '/*.wav')  # noqa
            case _ as err:
                raise ValueError(f"{err} is not supported. Expecting one of ['wav', 'png', ['wav', 'png']]")
        directories = [directories] if isinstance(directories, str) else directories
        for directory in directories:
            for f in fnames(directory):
                match self.file_extension(f):
                    case 'wav':
                        sig, rate = lb.load(f, sr=16000)
                        assert rate == 16000, f"sample rate must be 16kHz, got {rate} for {f}"
                        sig = sig[26000:]
                        sig = (sig - sig.mean()) / np.abs(sig).max()
                        sig = mfcc(y=sig, sr=rate)
                        self.wavs[f] = sig.T
                    case 'png':
                        # TODO: add image extraction
                        self.pngs = f
        if not self.pngs or not self.wavs:
            raise ValueError("Directory with train or(and) test samples does not exist")

    def get_wavs(self):
        return np.vstack(list(self.wavs.values()))

    # def pngs(self):
    #     return np.vstack(self.pngs.values())


def main():
    dataset = Dataset(['target_dev', 'target_train'])
    print(len(dataset.wavs))


if __name__ == "__main__":
    main()