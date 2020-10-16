import util.truthful_bidder as truthful_bidder



class DSP():
    def __init__(self, name, ctr_max_iter=10, max_bid_price=300):
        self.name = name
        self.ctr_max_iter = ctr_max_iter
        self.bidder = truthful_bidder.Truthful_bidder(max_iter=ctr_max_iter)
        self.max_bid_price = max_bid_price

    def train(self, x, y, z):
        _, alpha = self.bidder.fit(x, y, z)
        self.bidder.evaluate(x, y)



