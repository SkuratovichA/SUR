import os
from glob import glob
import numpy as np
import librosa as lb
from librosa.feature import mfcc
import speechbrain as sb
import torch

class Perturbator:
  '''Perturbator class, augments stuff'''
  def augment_data(self, data, sr):
    data = torch.tensor(data).unsqueeze(0)
    val = np.random.rand()
    _id = np.random.randint(11)
    aug_data = np.empty([1])
    try:

      if _id < 2:
        aug_data = self.__change_speed(data, sr)
      elif 1 < _id < 7:
        aug_data = self.__reverb(data, val)
      elif 6 < _id < 9:
        aug_data = self.__add_crowd_noise(data)
      elif 8 < _id < 11:
        aug_data = self.__add_noise(data, val)

    except:

      if _id < 6:
        aug_data, speed_up = self.__change_speed(data, sr)
      else: 
        aug_data = self.__add_noise(data, val)

    return aug_data.squeeze().numpy()

  def __change_speed(self, data, sr):
    '''
    Weight of this augmentation: 2
    '''
    perturbator = sb.processing.speech_augmentation.SpeedPerturb(orig_freq=sr, speeds=[90, 95, 105, 110])
    return perturbator(data)

  def __reverb(self, data, val):
    ''' 
    Weight of this augmentation: 5
    Val - random number, results in 3 cases:
    < 0.33 - 0.5 RIR scale factor
    < 0.67 - 1.0 RIR scale factor
    > 0.66 - 2.0 RIR scale factor
    '''

    scale_factor = 0.5
    if 0.33 < val < 0.67:
      scale_factor = 1
    elif val > 0.66:
      scale_factor = 2
    
    reverb = sb.processing.speech_augmentation.AddReverb('samples/rir_samples/rirs.csv', 
                                                      sorting='random',
                                                      rir_scale_factor=scale_factor)
    return reverb(data, torch.ones(1))

  def __add_crowd_noise(self, data):
    '''
    Weight of this augmentation: 2
    '''
    noisifier = sb.processing.speech_augmentation.AddNoise('./samples/noise_samples/noise.csv', normalize=True)
    return noisifier(data, torch.ones(1))

  def __add_noise(self, data, val):
    '''
    Weight of this augmentation: 2
    '''
    noise = np.random.randn(len(data))
    return data + val*100 * noise
    #augmented_data = augmented_data.astype(type(data[0])) tf is this?

class VoiceActivityDetector:

    def __init__(self):
        self.step = 160
        self.buffer_size = 160 
        self.buffer = np.array([],dtype=np.int16)
        self.out_buffer = np.array([],dtype=np.int16)
        self.n = 0
        self.VADthd = 0.
        self.VADn = 0.
        self.silence_counter = 0

    # Voice Activity Detection
    def vad(self, _frame):
        frame = np.array(_frame) ** 2.
        result = True
        threshold = 0.1 # adaptive threshold
        thd = np.min(frame) + np.ptp(frame) * threshold
        self.VADthd = (self.VADn * self.VADthd + thd) / float(self.VADn + 1.)
        self.VADn += 1.

        if np.mean(frame) <= self.VADthd:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

        if self.silence_counter > 20:
            result = False
        return result

    # Push new audio samples into the buffer.
    def add_samples(self, data):
        self.buffer = np.append(self.buffer, data)
        result = len(self.buffer) >= self.buffer_size
        return result

    # Pull a portion of the buffer to process
    # (pulled samples are deleted after being
    # processed
    def get_frame(self):
        window = self.buffer[:self.buffer_size]
        self.buffer = self.buffer[self.step:]
        return window

    # Adds new audio samples to the internal
    # buffer and process them
    def process(self, data):
        if self.add_samples(data):
            while len(self.buffer) >= self.buffer_size:
                # Framing
                window = self.get_frame()
                if self.vad(window):  # speech frame
                	self.out_buffer = np.append(self.out_buffer, window)

    def get_voice_samples(self):
        return self.out_buffer

class Dataset:
    def __init__(self, directories, extensions=None, aug = False):
        r"""
        Creates a dataset files in specified directories.

        :param directories: list of directories or a string representing a directory
        :param extensions: one of 'wav', 'png'. If None, both are included.
        :return: dictionary of type {'<person id>' : {'png' : filename.png, 'wav' : filename.wav}}
        """
        self.wavs = {}      # clear wavs
        self.wavsMfcc = {}  # mfcc wavs
        self.nonCutWavs = {}
        self.pngs = {}
        self.samples = {}
        self.dirfilter = lambda x: os.path.splitext(os.path.basename(x))  # 'smth/honza.wav' -> ['honza', 'wav']
        self.file_extension = lambda x: self.dirfilter(x)[1][1:]
        self.perturbator = Perturbator()

        extensions = [extensions] if isinstance(extensions, str) else extensions
        if extensions == 'wav':
          fnames = lambda d: glob(d + '/*.wav')  # noqa
        elif extensions == 'png':
          fnames = lambda d: glob(d + "/*.png")  # noqa
        elif extensions in ['wav', 'png'] or not extensions:
          fnames = lambda d: glob(d + "/*.png") + glob(d + '/*.wav')  # noqa
        else:
          raise ValueError(f"{extensions} is not supported. Expecting one of ['wav', 'png', ['wav', 'png']]")
        directories = [directories] if isinstance(directories, str) else directories
        for directory in directories:
          for f in fnames(directory):
            if self.file_extension(f) == 'wav':
              sig, rate = lb.load(f, sr=16000)
              self.nonCutWavs[f] = sig
              assert rate == 16000, f"sample rate must be 16kHz, got {rate} for {f}"
              sig = sig[26000:] # cut first 2 seconds
              VAD = VoiceActivityDetector()
              VAD.process(sig) # cut silence
              sig = VAD.get_voice_samples()
              if aug: # augment
                self.__augment_data(sig, f, rate)
              sig = (sig - sig.mean()) / np.abs(sig).max()
              self.wavs[f] = sig
              sig = mfcc(y=sig, sr=rate)
              sig = (sig - sig.mean()) / np.std(sig)
              self.wavsMfcc[f] = sig.T
            elif self.file_extension(f) == 'png':
              self.pngs = f
        if not self.pngs or not self.wavs:
          raise ValueError("Directory with train or(and) test samples does not exist")

    def __augment_data(self, data, filename, rate):
      aug_data = self.perturbator.augment_data(data, rate)
      aug_data = (aug_data - aug_data.mean()) / np.abs(aug_data).max()
      self.wavs[filename[:-4]+"-aug.wav"] = aug_data
      aug_data = mfcc(y=aug_data, sr=rate)
      self.wavsMfcc[filename[:-4]+"-aug.wav"] = aug_data.T


    def get_wavsMfcc(self):
        return np.vstack(list(self.wavsMfcc.values()))

    def get_wavs(self):
        return self.wavs

    # def pngs(self):
    #     return np.vstack(self.pngs.values())


def main():
    dataset = Dataset(['target_dev', 'target_train'])
    print(len(dataset.wavsMfcc))


if __name__ == "__main__":
    main()