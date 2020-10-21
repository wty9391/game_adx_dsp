import numpy as np
import sklearn.metrics
from sklearn import linear_model
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dsp import DSP
from util.util import Util, Chunk


class ADX:
    def __init__(self, dsp_number, ctr_max_iter=50, max_bid_price=300):
        self.dsp_number = dsp_number
        self.max_bid_price = max_bid_price
        self.dsp = [DSP(i, ctr_max_iter=ctr_max_iter, max_bid_price=300) for i in range(self.dsp_number)]
        self.ctr_model = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-6, verbose=0, n_jobs=4, max_iter=ctr_max_iter)
        self.ecpc = 0
        self.rp = 0
        self.highest_bids = Chunk(list(range(0, self.max_bid_price)))
        self.second_highest_bids = Chunk(list(range(0, self.max_bid_price)))
        self.writer = None

        self.br = 0
        self.imp = 0
        self.click = 0
        self.abort = 0
        self.abort_click = 0
        self.revenue = 0
        self.revenue_gain = 0

        self.auction_history = pd.DataFrame(
            columns=["round", "highest_bids", "second_highest_bids", "market_prices", "pctrs"])


    def train_dsp(self, x, y, z, indices, x_test, y_test, z_test):
        print('Training DSPs...')
        for i in range(self.dsp_number):
            print('--For DSP:{0:d}'.format(i))
            index = indices[i]
            self.dsp[i].train(x[index, :], y[index, :], z[index, :])
            print('eCPC:{0:.2f}'.format(self.dsp[i].bidder.eCPC))
            self.dsp[i].bidder.evaluate(x_test, y_test)

    def train_adx(self, x, y, z, x_test, y_test, z_test):
        print('Training ADX...')
        self.ctr_model.fit(x, y.ravel())
        self.ecpc = (np.sum(z, axis=0) / np.sum(y, axis=0))[0]
        auc = sklearn.metrics.roc_auc_score(y_test, self.ctr_model.predict_proba(x_test)[:, 1])
        print('Evaluation: classify auc is: {}%'.format(auc * 100))
        print('eCPC:{0:.2f}'.format(self.ecpc))

    def predict_ctr(self, bid_requests):
        return self.ctr_model.predict_proba(bid_requests)[:, 1].reshape((-1, 1))

    def play(self, name, bid_requests, clicks, origin_market_prices, dsp_total_budget, game_interval=1e4):
        (br_size, _) = np.shape(bid_requests)

        self.writer = SummaryWriter('runs/' + name)

        for i in range(self.dsp_number):
            self.dsp[i].play(dsp_total_budget / self.dsp_number, br_size)

        starts, ends = Util.generate_batch_index(br_size, game_interval)
        round = 0
        for start, end in list(zip(starts, ends)):
            print("=====Round: {0:d}=====".format(round))
            this_brs = bid_requests[start:end, :]
            this_clicks = clicks[start:end, :]

            (this_brs_size, _) = np.shape(this_brs)

            bids = np.concatenate([self.dsp[i].bid(this_brs) for i in range(self.dsp_number)], axis=1)
            rps = self.get_reserve_price(this_brs)

            is_not_abort, is_winner, market_prices, highest_bids, second_highest_bids = self.do_auction(bids, rps)

            self.highest_bids.append_data(highest_bids)
            self.second_highest_bids.append_data(second_highest_bids)

            pctr = self.predict_ctr(this_brs)
            df = pd.DataFrame([[round, highest_bids[i, 0], second_highest_bids[i, 0],
                                market_prices[i, 0], pctr[i, 0]] for i in range(this_brs_size)],
                              columns=["round", "highest_bids", "second_highest_bids", "market_prices", "pctrs"])

            self.auction_history = self.auction_history.append(df, ignore_index=True)

            # TODO: print report
            adx_report = PrettyTable(['bid_request', 'impression', 'abort', 'abort_proportion', 'revenue',
                                      'revenue_gain', 'abort_click', 'T_bid_request', 'T_impression', 'T_abort',
                                      'T_abort_proportion', 'T_revenue', 'T_revenue_gain', 'T_abort_click',
                                      'reserve_price'])
            dsp_report = PrettyTable(['index', 'budget', 'available_budget',
                                      'impression', 'click', 'cost',
                                      'T_impression', 'T_click', 'T_cost',
                                      'bidding_para'])

            T_is_abort = np.logical_not(is_not_abort)
            T_bid_request = this_brs_size
            T_impression = is_not_abort.sum()
            T_abort_num = T_is_abort.sum()
            T_revenue = market_prices[is_not_abort].sum()
            T_revenue_gain = T_revenue - second_highest_bids[is_not_abort].sum() - second_highest_bids[T_is_abort].sum()
            T_abort_click = this_clicks[T_is_abort].sum()

            self.br += T_bid_request
            self.imp += T_impression
            self.abort += T_abort_num
            self.revenue += T_revenue
            self.revenue_gain += T_revenue_gain
            self.abort_click += T_abort_click

            self.writer.add_scalar('adx/bid_request', self.br, round)
            self.writer.add_scalar('adx/impression', self.imp, round)
            self.writer.add_scalar('adx/abort', self.abort, round)
            self.writer.add_scalar('adx/abort_proportion', self.abort / self.br, round)
            self.writer.add_scalar('adx/revenue', self.revenue, round)
            self.writer.add_scalar('adx/revenue_gain', self.revenue_gain, round)
            self.writer.add_scalar('adx/abort_click', self.abort_click, round)

            self.writer.add_scalar('adx/T_bid_request', T_bid_request, round)
            self.writer.add_scalar('adx/T_impression', T_impression, round)
            self.writer.add_scalar('adx/T_abort', T_abort_num, round)
            self.writer.add_scalar('adx/T_abort_proportion', T_abort_num / T_bid_request, round)
            self.writer.add_scalar('adx/T_revenue', T_revenue, round)
            self.writer.add_scalar('adx/T_revenue_gain', T_revenue_gain, round)
            self.writer.add_scalar('adx/T_abort_click', T_abort_click, round)

            self.writer.add_scalar('adx/reserve_price', self.rp, round)

            adx_report.add_row([self.br, self.imp, self.abort, np.round(self.abort / self.br, 4), self.revenue,
                                self.revenue_gain, self.abort_click, T_bid_request, T_impression, T_abort_num,
                                np.round(T_abort_num / T_bid_request, 4), T_revenue, T_revenue_gain, T_abort_click,
                                self.rp])

            for i in range(self.dsp_number):
                self.dsp[i].bidding_result_notice(round, this_brs, market_prices, is_winner[:, i], this_clicks)

                T_imp = is_winner[:, i].sum()
                T_clk = this_clicks[is_winner[:, i]].sum()
                T_cost = market_prices[is_winner[:, i]].sum()

                self.writer.add_scalar('dsp/{0:d}/budget'.format(i), self.dsp[i].budget, round)
                self.writer.add_scalar('dsp/{0:d}/available_budget'.format(i), self.dsp[i].available_budget, round)
                self.writer.add_scalar('dsp/{0:d}/impression'.format(i), self.dsp[i].imp, round)
                self.writer.add_scalar('dsp/{0:d}/click'.format(i), self.dsp[i].click, round)
                self.writer.add_scalar('dsp/{0:d}/cost'.format(i), self.dsp[i].cost, round)
                self.writer.add_scalar('dsp/{0:d}/T_impression'.format(i), T_imp, round)
                self.writer.add_scalar('dsp/{0:d}/T_click'.format(i), T_clk, round)
                self.writer.add_scalar('dsp/{0:d}/T_cost'.format(i), T_cost, round)
                self.writer.add_scalar('dsp/{0:d}/bidding_para'.format(i), int(self.dsp[i].bidder.alpha * self.dsp[i].bidder.eCPC), round)

                dsp_report.add_row([self.dsp[i].name, self.dsp[i].budget, self.dsp[i].available_budget,
                                    self.dsp[i].imp, self.dsp[i].click, self.dsp[i].cost,
                                    T_imp, T_clk, T_cost,
                                    int(self.dsp[i].bidder.alpha * self.dsp[i].bidder.eCPC)])

                self.dsp[i].update_strategy(br_size-end)
            self.update_strategy(this_brs)

            print(adx_report)
            print(dsp_report)

            round += 1

        # end of the game

        def plot_boxplot(x, y, data, name):
            figure_width = 10
            figure_height = figure_width / 21 * 9
            f, ax = plt.subplots(1, 1)
            sns.boxplot(y=y, x=x, data=data, ax=ax, palette="colorblind", fliersize=0.5)
            if y == "pctrs":
                ax.set_ylim([0, 0.01])

            fig = plt.gcf()
            fig.set_size_inches(figure_width, figure_height)
            plt.tight_layout(pad=0, w_pad=0, h_pad=0, rect=(0.01, 0, 0.99, 1.0))

            self.writer.add_figure(name + '/' + y, f)
            plt.close(f)

        plot_boxplot("round", "highest_bids", self.auction_history, name="adx")
        plot_boxplot("round", "second_highest_bids", self.auction_history, name="adx")
        plot_boxplot("round", "market_prices", self.auction_history, name="adx")
        plot_boxplot("round", "pctrs", self.auction_history, name="adx")

        for dsp in self.dsp:
            name = "dsp/{0:d}".format(dsp.name)
            plot_boxplot("round", "market_prices", dsp.market_price_history, name=name)
            plot_boxplot("round", "pctrs", dsp.bidding_history, name=name)
            plot_boxplot("round", "bids", dsp.bidding_history, name=name)

        self.writer.close()

        return

    def get_reserve_price(self, bid_requests):
        (br_size, _) = np.shape(bid_requests)

        rps = np.zeros(shape=(br_size, 1)) + self.rp
        # rps = np.random.randint(low=1, high=300, size=(br_size, 1))
        # TODO: calculate optimal reserve prices

        return rps

    def update_strategy(self, bid_requests):
        # TODO update strategy
        (br_size, _) = np.shape(bid_requests)

        assert self.highest_bids.values == self.second_highest_bids.values

        def revenue_gain_expectation(rp):
            gain = 0
            risk = 0
            for b1 in range(rp, self.max_bid_price):
                gain += self.highest_bids.pdf[b1] * \
                    sum([self.second_highest_bids.pdf_normalized[b1][b2] * (rp - b2)for b2 in range(rp)])

            for b1 in range(rp):
                risk += self.highest_bids.pdf[b1] * \
                    sum([self.second_highest_bids.pdf_normalized[b1][b2] * b2 for b2 in range(b1)])

            # gain = (self.second_highest_bids.cdf[rp] * rp - self.second_highest_bids.integra_cdf[rp])\
            #        * (1 - self.highest_bids.cdf[rp])

            return gain - risk

        gain = [revenue_gain_expectation(i) for i in range(0, self.max_bid_price)]

        self.rp = int(np.ceil(gain.index(max(gain)) * (1 - self.abort / self.br)))

        # print("update ADX's reserve price as {0:d}".format(self.rp))

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






