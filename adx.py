import numpy as np
import sklearn.metrics
from sklearn import linear_model

import dsp

class ADX():
    def __init__(self, dsp_number=5, game_interval=1e5):
        self.dsp_number = dsp_number
        self.game_interval = game_interval
        self.dsp = [dsp.DSP(i) for i in range(self.dsp_number)]
        self.ctr_model = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-6, verbose=0, n_jobs=4, max_iter=10)
        self.ecpc = 0

    def train_dsp(self, x, y, z, indices):
        print('Training DSPs...')
        for i in range(self.dsp_number):
            print('--For DSP:{0:d}'.format(i))
            index = indices[i]
            self.dsp[i].train(x[index, :], y[index, :], z[index, :])
            print('eCPC:{0:.2f}'.format(self.dsp[i].bidder.alpha))

    def train_adx(self, x, y, z):
        print('Training ADX...')
        self.ctr_model.fit(x, y.ravel())
        self.ecpc = np.sum(z, axis=0) / np.sum(y, axis=0)
        auc = sklearn.metrics.roc_auc_score(y, self.ctr_model.predict_proba(x)[:, 1])
        print('Evaluation: classify auc is: {}%'.format(auc * 100))
        print('eCPC:{0:.2f}'.format(self.ecpc))

    def play(self, bid_requests):

        return