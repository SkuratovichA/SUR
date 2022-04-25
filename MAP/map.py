from dataset import Dataset
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from numpy.random import randint
from ikrlib import train_gmm, logpdf_gmm
from tqdm.notebook import trange, tqdm
from numpy import sum
import torch
import matplotlib.pyplot as plt
import IPython.display as ipd
from IPython.display import Audio
#!pip install speechbrain
import speechbrain as sb
from sklearn.mixture import BayesianGaussianMixture
import pickle

class Classifier:
    
    def __init__(self, target_train='../dataset/target_train', 
                    non_target_train='../dataset/non_target_train', 
                    target_dev='../dataset/target_dev', 
                    non_target_dev='../dataset/non_target_dev'):
        # train datasets
        self.train_d = Dataset(directories=target_train, aug=True)
        self.train_nd = Dataset(directories=non_target_train, aug=True)
        # test datasets;
        self.test_d = Dataset(directories=target_dev) 
        self.test_nd = Dataset(directories=non_target_dev)

        self.train_t = self.train_d.get_wavsMfcc()
        self.train_n = self.train_nd.get_wavsMfcc()

        X_target = self.train_t
        X_non_target = self.train_n

        self.bgmm_target = BayesianGaussianMixture(n_components=7,  random_state=69, init_params='random', max_iter=2000).fit(X_target)
        self.bgmm_non_target = BayesianGaussianMixture(n_components=9, random_state=42, init_params='random', max_iter=2000).fit(X_non_target)    

        self.evaluate()
        #self.print_score()

    def save(self, bgm_t_file = '../models/bgm_target_model.pkl', bgm_n_file = '../models/bgm_n_target_model.pkl'):
        ''' Save target bgmm model '''
        with open(bgm_t_file, 'wb') as file:
            pickle.dump(self.bgmm_target, file)
        ''' Save non-target bgmm model '''
        with open(bgm_n_file, 'wb') as file:
            pickle.dump(self.bgmm_non_target, file)

    def load(self, bgm_t_file = '../models/bgm_target_model.pkl', bgm_n_file = '../models/bgm_n_target_model.pkl'):
        ''' Load target bgmm model '''
        with open(bgm_t_file, 'rb') as file:
            self.bgmm_target = pickle.load(file)
        ''' Load target bgmm model '''
        with open(bgm_n_file, 'rb') as file:
            self.bgmm_non_target = pickle.load(file)

    def print_res(self, scores, printMe=True):
        if printMe:
            print("ONE CLASSIFIER FOR 2 DIFFERENT SETS")
            print("MAP CLASSIFIER CLASSIFIES TARGET AND NON-TARGET")
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

    def evaluate(self):
        
        P_t, P_n = 0.5, 0.5
        # change test to some testdir 
        for _, test in enumerate(self.test_nd.wavsMfcc):
            ll_t = self.bgmm_target.score_samples(test)
            ll_n = self.bgmm_non_target.score_samples(test)
            print(test, ":", int(sum(ll_t) < sum(ll_n)), f"{(sum(ll_t) - sum(ll_n)):.2f}    {int((sum(ll_t) - sum(ll_n)) > 0)}")

    def print_score(self):

        P_t = 0.5
        P_n = 1.0-P_t

        score_for_target_testset = []
        for tst in self.test_d.wavsMfcc.values(): # non target
            ll_t = self.bgmm_target.score_samples(tst) # target
            ll_n = self.bgmm_non_target.score_samples(tst) # non target
            score_for_target_testset.append(sum(ll_t) - sum(ll_n)) # prob target > prob non target
            
        score_for_non_target_testset = []
        for tst in self.test_nd.wavsMfcc.values(): # non target
            ll_t = self.bgmm_target.score_samples(tst) # target
            ll_n = self.bgmm_non_target.score_samples(tst) # non target
            #print(int(sum(ll_t) < sum(ll_n)))
            #print(f"{(sum(ll_t) - sum(ll_n)):.2f}    {int((sum(ll_t) - sum(ll_n)) > 0)}")
            score_for_non_target_testset.append(sum(ll_t) - sum(ll_n)) # prob target > prob non target

        self.print_res([score_for_target_testset, score_for_non_target_testset])


def main():
    
    classifier = Classifier()

main()
