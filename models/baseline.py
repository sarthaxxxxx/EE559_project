import numpy as np


class NearestMeans:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def euclidean_distance(x, y):
        return np.linalg.norm(x - y, axis = 1)

    def train(self):
        class_means = [
            np.mean(self.x[self.y == c], axis=0) for c in np.unique(self.y)
        ]  
        return np.array(class_means)

    def predict(self, class_means, x):
        dist = []
        for mean in class_means:
            dist.append(self.euclidean_distance(x, mean))
        class_idx = np.argmin(dist, axis = 0)
        return np.array([np.unique(self.y)[idx] for idx in class_idx])

    def runner(self, val_x, val_y):
        class_means = self.train()
        pred = self.predict(class_means, val_x)
        return pred
