import numpy as np

import util.truthful_bidder as truthful_bidder

class DSP():
    def __init__(self, name, budget=1e5, ctr_max_iter=10, max_bid_price=300):
        self.name = name
        self.budget = budget
        self.ctr_max_iter = ctr_max_iter
        self.bidder = truthful_bidder.Truthful_bidder(max_iter=ctr_max_iter)
        self.max_bid_price = max_bid_price

        self.br = 0
        self.imp = 0
        self.click = 0
        self.cost = 0

    def train(self, x, y, z):
        _, alpha = self.bidder.fit(x, y, z)
        self.bidder.evaluate(x, y)

    def bid(self, bid_requests):
        (br_size, _) = np.shape(bid_requests)

        # bids = np.zeros(shape=(br_size, 1))
        bids = np.random.randint(low=1, high=300, size=(br_size, 1))
        # TODO: calculate optimal bid prices


        return bids

    def update_strategy(self, bid_requests, bids, is_winner):
        # TODO update strategy

        return