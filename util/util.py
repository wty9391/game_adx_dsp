import sys
import math
import numpy as np

eps = sys.float_info.epsilon


class Util:
    def __init__(self):
        pass

    @staticmethod
    def generate_batch_index(size, batch_size=4096):
        starts = [int(i * batch_size) for i in range(int(math.ceil(size / batch_size)))]
        ends = [int(i * batch_size) for i in range(1, int(math.ceil(size / batch_size)))]
        ends.append(int(size))

        return starts, ends

    @staticmethod
    def cdf_to_pdf(cdf):
        cdf = cdf.copy()
        cdf.insert(0, 0.0 + eps)
        pdf = [cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else 1e-6 for i in range(len(cdf) - 1)]
        # normalize
        pdf_sum = sum(pdf)

        return [p / pdf_sum for p in pdf]

    @staticmethod
    def pdf_to_cdf(pdf):
        # normalize
        pdf_sum = sum(pdf)
        pdf = [pdf[i] / pdf_sum if pdf[i] > 0 else eps for i in range(len(pdf))]

        cdf = [sum(pdf[0:i+1]) for i in range(len(pdf))]
        return cdf


class Chunk:
    def __init__(self, values):
        self.data = np.asarray([]).reshape((-1, 1))
        self.values = []
        self.pdf = []
        self.pdf_normalized = []
        self.cdf = []
        self.expectation = 0
        self.integra_pdf = []
        self.integra_cdf = []

        self.append_data(values)

    def update_data(self, data):
        self.__init__()
        self.append_data(data)

    def append_data(self, data, drop_rate=0.25):
        """

        :param drop_rate:
        :param data: could be array or list
        :return:
        """
        data = np.asarray(data).reshape((-1, 1))

        mask = np.random.choice([False, True], len(self.data), p=[drop_rate, 1 - drop_rate])

        self.data = np.vstack((self.data[mask], data))
        (unique, counts) = np.unique(self.data, return_counts=True)  # the unique has been sorted

        diff = np.asarray(list(set(self.values) - set(unique))).reshape((-1, 1))
        self.data = np.vstack((self.data, diff))
        (unique, counts) = np.unique(self.data, return_counts=True)

        assert all(unique[i] <= unique[i+1] for i in range(len(unique)-1))

        self.values = unique.tolist()
        self.cdf = Util.pdf_to_cdf(counts)
        self.pdf = Util.cdf_to_pdf(self.cdf)

        self.expectation = np.mean(self.data)
        self.integra_pdf = []
        self.integra_cdf = []
        for i in range(len(self.values)):
            self.integra_pdf.append(self.pdf[i] * self.values[i])
            self.pdf_normalized.append([self.pdf[j] / self.cdf[i] for j in range(i+1)])
        self.integra_cdf = [sum(self.integra_pdf[0:i+1]) for i in range(len(self.integra_pdf))]

        assert len(self.values) == len(self.pdf) == len(self.cdf) == \
               len(self.integra_pdf) == len(self.integra_cdf)\
            , "the length of values, pdf and cdf must be equal"

