import sys
import math

eps = sys.float_info.epsilon

class Util():
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
        pdf = [cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else 1e-6 for i in range(len(cdf) - 1)]
        pdf.insert(0, 0.0 + eps)
        # normalize
        pdf_sum = sum(pdf)

        return [p / pdf_sum for p in pdf]

    @staticmethod
    def pdf_to_cdf(pdf):
        # normalize
        pdf_sum = sum(pdf)
        pdf = [pdf[i] / pdf_sum if pdf[i] > 0 else 1e-6 for i in range(len(pdf))]

        cdf = [sum(pdf[0:i]) for i in range(1, len(pdf))]
        cdf.append(1.0)
        return cdf
