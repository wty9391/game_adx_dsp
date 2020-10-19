import numpy as np
import sklearn.metrics
from sklearn import linear_model
from prettytable import PrettyTable

from dsp import DSP
from util.util import Util, Chunk


class ADX():
    def __init__(self, dsp_number, ctr_max_iter=50, max_bid_price=300):
        self.dsp_number = dsp_number
        self.max_bid_price = max_bid_price
        self.dsp = [DSP(i, ctr_max_iter=ctr_max_iter, max_bid_price=300) for i in range(self.dsp_number)]
        self.ctr_model = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-6, verbose=0, n_jobs=4, max_iter=ctr_max_iter)
        self.ecpc = 0
        self.rp = 0
        self.highest_bids = Chunk()
        self.second_highest_bids = Chunk()

    def train_dsp(self, x, y, z, indices):
        print('Training DSPs...')
        for i in range(self.dsp_number):
            print('--For DSP:{0:d}'.format(i))
            index = indices[i]
            self.dsp[i].train(x[index, :], y[index, :], z[index, :])
            print('eCPC:{0:.2f}'.format(self.dsp[i].bidder.eCPC))

    def train_adx(self, x, y, z):
        print('Training ADX...')
        self.ctr_model.fit(x, y.ravel())
        self.ecpc = (np.sum(z, axis=0) / np.sum(y, axis=0))[0]
        auc = sklearn.metrics.roc_auc_score(y, self.ctr_model.predict_proba(x)[:, 1])
        print('Evaluation: classify auc is: {}%'.format(auc * 100))
        print('eCPC:{0:.2f}'.format(self.ecpc))

    def play(self, bid_requests, clicks, origin_market_prices, dsp_total_budget, game_interval=1e4):
        (br_size, _) = np.shape(bid_requests)
        for i in range(self.dsp_number):
            self.dsp[i].play(dsp_total_budget / self.dsp_number, br_size)

        starts, ends = Util.generate_batch_index(br_size, game_interval)
        round = 0
        for start, end in list(zip(starts, ends)):
            print("=====Round: {0:d}=====".format(round))
            this_brs = bid_requests[start:end, :]
            this_clicks = clicks[start:end, :]

            bids = np.concatenate([self.dsp[i].bid(this_brs) for i in range(self.dsp_number)], axis=1)
            rps = self.get_reserve_price(this_brs)

            is_not_abort, is_winner, market_prices, highest_bids, second_highest_bids = self.do_auction(bids, rps)

            self.update_strategy(this_brs, is_not_abort, highest_bids, second_highest_bids)

            # TODO: print report
            adx_repost = PrettyTable([''])
            dsp_report = PrettyTable(['index', 'budget', 'available_budget',
                                      'impression', 'click', 'cost',
                                      'T_impression', 'T_click', 'T_cost',
                                      'para'])

            for i in range(self.dsp_number):
                self.dsp[i].bidding_result_notice(round, this_brs, market_prices, is_winner[:, i], this_clicks)
                self.dsp[i].update_strategy(br_size-end)

                T_imp = is_winner[:, i].sum()
                T_clk = this_clicks[is_winner[:, i]].sum()
                T_cost = market_prices[is_winner[:, i]].sum()

                dsp_report.add_row([self.dsp[i].name, self.dsp[i].budget, self.dsp[i].available_budget,
                                    self.dsp[i].imp, self.dsp[i].click, self.dsp[i].cost,
                                    T_imp, T_clk, T_cost,
                                    int(self.dsp[i].bidder.alpha * self.dsp[i].bidder.eCPC)])
            print(dsp_report)
            round += 1
        return

    def get_reserve_price(self, bid_requests):
        (br_size, _) = np.shape(bid_requests)

        rps = np.zeros(shape=(br_size, 1)) + self.rp
        # rps = np.random.randint(low=1, high=300, size=(br_size, 1))
        # TODO: calculate optimal reserve prices

        return rps

    def update_strategy(self, bid_requests, is_not_abort, highest_bids, second_highest_bids):
        # TODO update strategy
        reserve_price_candidates = list(range(0, self.max_bid_price))
        self.highest_bids.update_data(reserve_price_candidates)
        self.highest_bids.append_data(highest_bids)

        self.second_highest_bids.update_data(reserve_price_candidates)
        self.second_highest_bids.append_data(second_highest_bids)

        assert self.highest_bids.values == self.second_highest_bids.values

        def revenue_gain_expectation(rp):
            gain = (self.second_highest_bids.cdf[rp] * rp - self.second_highest_bids.integra_cdf[rp])\
                   * (1 - self.highest_bids.cdf[rp])
            risk = sum([self.highest_bids.pdf[i]*self.second_highest_bids.integra_cdf_normalized[i] for i in range(rp)])
            return gain - risk

        gain = [revenue_gain_expectation(i) for i in range(0, self.max_bid_price)]
        self.rp = gain.index(max(gain))

        print("update ADX's reserve price as {0:d}".format(self.rp))

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
        highest_bids = np.max(bids, axis=1).reshape((-1, 1))

        is_abort = rps >= highest_bids
        is_not_abort = np.logical_not(is_abort)

        # In case of multiple occurrences of the maximum values,
        # the indices corresponding to the first occurrence are returned.
        winner_index = np.argmax(bids, axis=1)

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

        return is_not_abort, is_winner, market_prices, highest_bids, second_highest_bids






