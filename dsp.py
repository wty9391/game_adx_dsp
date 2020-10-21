import numpy as np
import pandas as pd

import util.truthful_bidder as truthful_bidder
from util.util import Util, Chunk

class DSP():
    def __init__(self, name, ctr_max_iter=10, max_bid_price=300):
        self.name = name
        self.ctr_max_iter = ctr_max_iter
        self.bidder = truthful_bidder.Truthful_bidder(max_iter=ctr_max_iter)
        self.max_bid_price = max_bid_price
        self.market_price = Chunk(list(range(0, self.max_bid_price)))

        self.budget = 0
        self.available_budget = 0
        self.br = 0
        self.imp = 0
        self.click = 0
        self.cost = 0

        self.last_br = None  # used to update strategy

        self.bidding_history = pd.DataFrame(columns=["round", "pctrs", "bids"])
        self.market_price_history = pd.DataFrame(columns=["round", "market_prices"])

    def train(self, x, y, z):
        _, alpha = self.bidder.fit(x, y, z)
        # self.market_price.append_data(z)

        self.last_br = x

    def bid(self, bid_requests):
        (br_size, _) = np.shape(bid_requests)

        # TODO: calculate optimal bid prices
        # bids = np.zeros(shape=(br_size, 1))
        # bids = np.random.randint(low=1, high=300, size=(br_size, 1))
        bids = self.bidder.bid(bid_requests).reshape((-1, 1)).astype(np.int)
        # noise = np.random.randint(low=95, high=105, size=(br_size, 1)) / 100
        # bids = (bids * noise).astype(np.int)
        bids[bids >= self.max_bid_price] = self.max_bid_price - 1
        bids[bids <= 0] = 1

        self.last_br = bid_requests

        return bids

    def predict_ctr(self, bid_requests):
        return self.bidder.model.predict_proba(bid_requests)[:, 1].reshape((-1, 1))

    def play(self, budget, available_br_number):
        self.budget = int(budget)
        self.available_budget = self.budget
        self.br = 0
        self.imp = 0
        self.click = 0
        self.cost = 0

        self.update_strategy(available_br_number)

    def bidding_result_notice(self, round, bid_requests, market_prices, is_winner, click):
        (br_size, _) = np.shape(bid_requests)
        cost = market_prices[is_winner].sum()
        self.cost += cost
        self.available_budget = self.budget - self.cost

        self.br += br_size
        self.imp += is_winner.sum()
        self.click += click[is_winner].sum()

        self.market_price.append_data(market_prices[is_winner])

        pctr = self.predict_ctr(bid_requests)
        bids = self.bid(bid_requests)
        df = pd.DataFrame([[round, pctr[i, 0], bids[i, 0]] for i in range(br_size)],
                          columns=["round", "pctrs", "bids"])
        self.bidding_history = self.bidding_history.append(df, ignore_index=True)

        df = pd.DataFrame([[round, market_prices[i, 0]] for i in range(is_winner.sum())],
                          columns=["round", "market_prices"])
        self.market_price_history = self.market_price_history.append(df, ignore_index=True)

    def update_strategy(self, available_br_number, max_iteration=10, converge_threshold=0.01):
        # TODO update strategy
        # print("update DSP: {0:d}'s bidding strategy".format(self.name))
        br = self.last_br
        (br_size, _) = np.shape(br)
        # print("available budget: {0:.0f}".format(self.available_budget))

        if self.available_budget <= 0:
            # print("No available budget")
            self.bidder.alpha = 0
        else:
            for i in range(max_iteration):
                # print("== iteration ", i, " ==")
                bids = self.bid(br)
                e_cost = sum([self.market_price.integra_cdf[bids[i, 0]] for i in range(br_size)])
                e_cost = e_cost / br_size * (available_br_number + 100)  # incase of zero
                deviation = (e_cost - self.available_budget) / self.available_budget
                # print("current alpha is {0:.5f}, current E-cost is {1:.2f}, relative deviation is {2:.2f}%"
                # .format(self.bidder.alpha, e_cost, deviation*100))

                if abs(deviation) < converge_threshold:
                    # print("=converged=")
                    break
                else:
                    self.bidder.alpha = self.bidder.alpha * self.available_budget / e_cost
            # print("Updated bidding parameter is {0:.2f}".format(self.bidder.alpha * self.bidder.eCPC))

