# File: map.py
# Author: Tikhonov Maksim <xtikho00@vutbr.cz>
# Date: 27.4.2022, 3.42 AM

import dataset as data
import numpy as np
from numpy import sum
from MAP.dataset import Dataset
from MAP.dataset import VoiceActivityDetector
from sklearn.mixture import BayesianGaussianMixture
import os
import librosa as lb
import pickle
from librosa.feature import mfcc

class Classifier:
    
    """ def __init__(self, target_train='../dataset/target_train', 
                    non_target_train='../dataset/non_target_train', 
                    target_dev='../dataset/target_dev', 
                    non_target_dev='../dataset/non_target_dev'): """
    def __init__(self, hparams):
        self.hparams = hparams
        # CREATING DATASETS
        if hparams["train"]:
            self.train_d = Dataset(directories=self.hparams["dataset_dir"]["target"], aug=True)
            self.train_nd = Dataset(directories=self.hparams["dataset_dir"]["non_target"], aug=True)
            self.train_t = self.train_d.get_wavsMfcc()
            self.train_n = self.train_nd.get_wavsMfcc()

    def train(self):
        self.bgmm_target = BayesianGaussianMixture(n_components=1,  random_state=69, init_params='random', max_iter=2000).fit(self.train_t)
        self.bgmm_non_target = BayesianGaussianMixture(n_components=1, random_state=42, init_params='random', max_iter=2000).fit(self.train_n)

    def save(self):
        ''' Save target bgmm model '''
        with open(os.path.join(self.hparams["model_dir"], self.hparams["model_name"]["target"]), 'wb') as file:
            pickle.dump(self.bgmm_target, file)
        ''' Save non-target bgmm model '''
        with open(os.path.join(self.hparams["model_dir"], self.hparams["model_name"]["non_target"]), 'wb') as file:
            pickle.dump(self.bgmm_non_target, file)

    def load(self):
        ''' Load target bgmm model '''
        with open(os.path.join(self.hparams["model_dir"], self.hparams["model_name"]["target"]), 'rb') as file:
            self.bgmm_target = pickle.load(file)
        ''' Load target bgmm model '''
        with open(os.path.join(self.hparams["model_dir"], self.hparams["model_name"]["non_target"]), 'rb') as file:
            self.bgmm_non_target = pickle.load(file)

    def print_res(self, scores, printMe=True):
        if printMe:
            print("RESULTS ARE FOR TARGET SET AND NON TARGET SET ")
        ss = [("target", lambda x: x > 0) , ("non target", lambda x: x < 0)]
        def correctness(score, typ="", foo=None):
            n_score = np.array(score)
            if printMe:
                print(f"{typ}:")
                print(f"\tSum: {sum(n_score)}")
                print(f"\tMean: {n_score.mean()}")
                print(f"\tVar: {n_score.std()}")
                print(f"\tMin: {n_score.min()}")
                print(f"\tMax: {n_score.max()}")

            correctness = len(np.array(n_score[np.where(foo(n_score))])) / len(n_score)
            if printMe:
                print(f"\t\tOverall correctness: {correctness*100}%")
            return correctness, n_score.std(), n_score.mean()

        corrs = [0,0]
        stdivs = [0,0]
        means = [0,0]
        for i, name in enumerate(scores):
            if printMe:
                print(f"{ss[i][0]} predictions: {name}")
            c, stdiv, mean = correctness(name, typ=ss[i][0], foo=ss[i][1])
            corrs[i] = c
            stdivs[i] = stdiv
            means[i] = mean

        maximize_me = (corrs[0]*corrs[1])**30 * (abs(means[0] - means[1])/(stdivs[0]+stdivs[1]))
        print(f"MAXIMIZE ME: {maximize_me}")
        return maximize_me

    def evaluate(self, filename):
        # get file
        sig, rate = lb.load(filename, sr=16000)
              
        sig = sig[26000:] # cut first 2 seconds
        VAD = VoiceActivityDetector()
        sig = VAD.process(sig) # cut silence
        # normalise both signal and mfc coefficients
        sig = (sig - sig.mean()) / np.abs(sig).max()
        sig = mfcc(y=sig, sr=rate)
        sig = ((sig - sig.mean()) / np.std(sig)).T
        # evaluate
        ll_t = self.bgmm_target.score_samples(sig)
        ll_n = self.bgmm_non_target.score_samples(sig)
        return sum(ll_t) - sum(ll_n), int((sum(ll_t) - sum(ll_n)) > 0)

    def evaluateIter(self):
        # change test to some testdir
        for testname, wav in self.test_nd.wavsMfcc.items():
            ll_t = self.bgmm_target.score_samples(self.test_nd.wavsMfcc[testname])
            ll_n = self.bgmm_non_target.score_samples(self.test_nd.wavsMfcc[testname])
            print(testname.split('/')[-1], f"{(sum(ll_t) - sum(ll_n)):.2f}    {int((sum(ll_n) - sum(ll_t)) > 0)}")#, file=self.hparams["eval_dir"]+"/"+self.params["eval_out"])

    def print_score(self):

        score_for_target_testset = []
        for tst in self.test_d.wavsMfcc.values(): # non target
            ll_t = self.bgmm_target.score_samples(tst) # target
            ll_n = self.bgmm_non_target.score_samples(tst) # non target
            score_for_target_testset.append(sum(ll_t) - sum(ll_n)) # prob target > prob non target
            
        score_for_non_target_testset = []
        for tst in self.test_nd.wavsMfcc.values(): # non target
            ll_t = self.bgmm_target.score_samples(tst) # target
            ll_n = self.bgmm_non_target.score_samples(tst) # non target
            score_for_non_target_testset.append(sum(ll_t) - sum(ll_n)) # prob target > prob non target

        self.print_res([score_for_target_testset, score_for_non_target_testset])


def main(hparams):
    
    return Classifier(hparams)
