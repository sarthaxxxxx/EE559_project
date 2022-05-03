import math
import numpy as np


class Trivial:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def classwise_data(self):
        classes = np.unique(self.y)
        return len(self.x[classes[0] == self.y]), \
        len(self.x[classes[1] == self.y])

    def fit(self, test_y):
        S1_prob = self.classwise_data()[0] / len(self.y)
        S2_prob = self.classwise_data()[1] / len(self.y)
        total_zeros = [0] * math.ceil(S1_prob * len(test_y))
        total_ones = [1] * math.ceil(S2_prob * len(test_y))
        preds = np.concatenate((total_zeros, total_ones))[:len(test_y)]
        assert len(preds) == len(test_y), 'Predictions and data are not the same length'
        return preds
