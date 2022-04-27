#from dataset import Dataset
import dataset as data
import numpy as np
from numpy import sum
from MAP.dataset import Dataset
from sklearn.mixture import BayesianGaussianMixture
import os
import pickle

class Classifier:
    
    """ def __init__(self, target_train='../dataset/target_train', 
                    non_target_train='../dataset/non_target_train', 
                    target_dev='../dataset/target_dev', 
                    non_target_dev='../dataset/non_target_dev'): """
    def __init__(self, hparams):
        self.hparams = hparams
        # train datasets
        print("CREATING DATASETS...")
        self.train_d = Dataset(directories=self.hparams["dataset_dir"]["target"], aug=True)
        self.train_nd = Dataset(directories=self.hparams["dataset_dir"]["non_target"], aug=True)
        #remove before eval
        #self.eval = data.Dataset(directiories=self.hparams["eval_dataset"])
        self.test_d = Dataset(directories=self.hparams["dev_dataset"]["target"]) 
        self.test_nd = Dataset(directories=self.hparams["dev_dataset"]["non_target"])
        if hparams["train"]:
            self.train_t = self.train_d.get_wavsMfcc()
            self.train_n = self.train_nd.get_wavsMfcc()
            print("TRAINING...")
            self.train()
        else:
            self.load()
        print("EVALUATING...")
        self.evaluateIter()
        print("SAVING...")
        self.save()

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
        with open(os.path.join(self.hparams["model_dir"], self.hparams["model_name"]["non_target"]), 'rb') as file:
            self.bgmm_target = pickle.load(file)
        ''' Load target bgmm model '''
        with open(os.path.join(self.hparams["model_dir"], self.hparams["model_name"]["target"]), 'rb') as file:
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
        ll_t = self.bgmm_target.score_samples(self.eval.wavsMfcc[filename])
        ll_n = self.bgmm_non_target.score_samples(self.eval.wavsMfcc[filename])
        return f"{(sum(ll_t) - sum(ll_n)):.2f}", int((sum(ll_t) - sum(ll_n)) > 0)

    def evaluateIter(self):
        # change test to some testdir
        print(self.test_d.wavsMfcc)
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

""" 
hparams = { "train": False, # if not true -- load
            "eval": True,
            "model_dir": "./models",
            "dataset_dir": {"non_target": "./dataset/non_target_train", "target": "./dataset/target_train"},
            "dev_dataset" : {"non_target": "./dataset/non_target_dev", "target": "./dataset/target_dev"},

            "model_name": {"target": "bgmm_target.pkl", "non_target": "bgmm_non_target.pkl"},
            "eval_dir": "./eval",
            "eval_out": "output",
            "GPU": 0,
            "root_dir": ".",
            "wandb_entity": "skuratovich"}

main(hparams)
 """