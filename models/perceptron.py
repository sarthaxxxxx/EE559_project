import numpy as np

np.random.seed(0)

class Perceptron:
    def _init_(self):
        self.lr = 0.1
        self.n_iters = 1000

    def _init_weights(self, dims):
        return np.ones(dims) * 0.1

    @staticmethod
    def _indicator(op):
        # return 1 if g(w) <= 0, else return 0
        return 1 if op <= 0 else 0
    
    def _predict(self, w, x):
        # compute w.x
        return np.dot(w, x) 

    def _fit(self, x, y):   
        # init the weights
        weights = self._init_weights(x.shape[1])
        # obtain z_n for reflecion of the data
        z = np.array([1 if yi == 0 else -1 for yi in y])

        iters = 0
        J, w = [], []
        decision = False

        while iters <= 1000 and not decision:
            misclassified, J_w = 0, 0
            for idx in range(len(x)):
                # compute g(w) = w*z_n*x_n
                op = self._predict(weights, x[idx]) * z[idx]
                # if g(w) <= 0, misclassified
                if self._indicator(op) == 1:
                    weights += 0.01 * x[idx] * z[idx]
                    J_w += op
                    misclassified += 1
                else:
                    J_w += 0
                    weights = weights
            if misclassified == 0:
                # if no misclassified data, stop
                print('data is linearly separable')
                decision = True
                return -J_w, weights
            if iters >= 950:
                J.append(-J_w)
                w.append(weights)
            iters += 1
        optimal_weights = w[J.index(min(J))]
        return min(J), optimal_weights

    def _classify(self, x, w):  
        preds = [1 if self._predict(x[idx], w) <= 0 \
                else 0 for idx in range(len(x))]
        return np.array(preds)