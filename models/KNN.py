import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math


class KNN:
    def __init__(self, x, y, test_x, test_y, k = 5):
        self.x = x
        self.y = y
        self.k = k
        self.test_x = test_x
        self.test_y = test_y
        
    @staticmethod
    def euclidean_distance(data_v, data_t):
        squared_differences = [
            (data_v[idx] - data_t[idx])**2
            for idx in range(data_v.shape[0])
        ]
        return math.sqrt(sum(squared_differences))

    def knn_label(self, test_x):
        label_counter = {}
        classes = np.unique(self.y)
        for idx in classes: label_counter[idx] = 0
        min_indices = self.knn(test_x)
        for idx in range(min_indices.shape[0]):
            label_counter[self.y[min_indices[idx]]] += 1
        predicted_label = max(label_counter, key = label_counter.get)
        return predicted_label
        
    def knn(self, test_x):
        distances = [
            self.euclidean_distance(test_x, self.x[idx])
            for idx in range(self.x.shape[0])
        ]
        return np.array(distances).argsort()[:self.k]

    def pred_label_using_knn(self):
        predicted_label = [
            self.knn_label(self.test_x[idx])
            for idx in range(self.test_x.shape[0])
        ]
        return np.array(predicted_label)

    def calculate_accuracy(self, pred_labels):
        correct_pred = np.sum(pred_labels == self.test_y)
        acc = correct_pred / self.test_y.shape[0]
        return acc

    def fit(self):
        return self.pred_label_using_knn()
