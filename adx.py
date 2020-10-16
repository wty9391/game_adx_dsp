import numpy as np
import sklearn.metrics
from sklearn import linear_model

from dsp import DSP
from util.util import Util


class ADX():
    def __init__(self, dsp_number=5, ctr_max_iter=50):
        self.dsp_number = dsp_number
        self.dsp = [DSP(i, ctr_max_iter=ctr_max_iter) for i in range(self.dsp_number)]
        self.ctr_model = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-6, verbose=0, n_jobs=4, max_iter=ctr_max_iter)
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
        self.ecpc = (np.sum(z, axis=0) / np.sum(y, axis=0))[0]
        auc = sklearn.metrics.roc_auc_score(y, self.ctr_model.predict_proba(x)[:, 1])
        print('Evaluation: classify auc is: {}%'.format(auc * 100))
        print('eCPC:{0:.2f}'.format(self.ecpc))

    def play(self, bid_requests, clicks, market_prices, game_interval=1e4):
        (br_size, _) = np.shape(bid_requests)
        starts, ends = Util.generate_batch_index(br_size, game_interval)
        for start, end in list(zip(starts, ends)):
            this_brs = bid_requests[start:end, :]
            bids = np.concatenate([self.dsp[i].bid(this_brs) for i in range(self.dsp_number)], axis=1)
            rps = self.get_reserve_price(this_brs)
            is_not_abort, is_winner, market_prices, second_highest_bids = self.do_auction(bids, rps)

        return

    def get_reserve_price(self, bid_requests):
        (br_size, _) = np.shape(bid_requests)

        # rps = np.zeros(shape=(br_size, 1))
        rps = np.random.randint(low=1, high=300, size=(br_size, 1))
        # TODO: calculate optimal reserve prices

        return rps

    def update_strategy(self, bid_requests, bids, is_not_abort, second_highest_bids):
        # TODO update strategy

        return

    def do_auction(self, bids, rps):
        """
        Do auctions abort or not?
        Who are the winners of unabort auctions?
        What are the market prices of unabort auctions?
        What are 2nd highest bid prices of all auctions?

        :param bids: bid_size * dsp_number np.array
        :param rps:
        :return:
        """
        (bid_size, dsp_number) = np.shape(bids)
        max_bids = np.max(bids, axis=1).reshape((-1, 1))

        is_abort = rps >= max_bids
        is_not_abort = np.logical_not(is_abort)

        winner_index = np.argmax(bids, axis=1)
        # bid_winner_index = np.stack((np.arange(bid_size), winner_index), axis=1)

        is_winner = np.zeros_like(bids).astype(np.bool)
        is_winner[np.arange(bid_size), winner_index] = True
        is_winner[is_abort.reshape((-1,)), :] = False
        assert np.sum(is_winner) == np.sum(is_not_abort), \
            "the number of winners [{0:d}] must be equal to unabort auctions [{1:d}]"\
                .format(np.sum(is_winner), np.sum(is_not_abort))

        bids_copy = np.copy(bids)
        bids_copy[np.arange(bid_size), winner_index] = 0  # remove the highest bids
        second_highest_bids = np.max(bids_copy, axis=1).reshape((-1, 1))

        augment_bids = np.hstack((bids, rps))
        augment_bids[np.arange(bid_size), winner_index] = 0  # remove the highest bids
        market_prices = np.max(augment_bids, axis=1).reshape((-1, 1))  # only available for unabort auctions

        return is_not_abort, is_winner, market_prices, second_highest_bids






