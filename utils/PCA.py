import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, train, components):
        self.train_x = train
        self.components = components

    def _covariance(self):
        cov = np.cov(self.train_x.T)
        return cov

    def _eigen_vectors(self):
        cov = self._covariance()
        if np.equal(cov.transpose(), cov).all() and cov.shape[0] == cov.shape[1]:
            print('Covariance matrix is square and symmetric!!')
            print('Eigenvectors are orthogonal!!')
            def sorted_eig(cov):
                eig_vals, eig_vecs = np.linalg.eig(cov)
                idx = eig_vals.argsort()[::-1]
                eig_vals = eig_vals[idx]
                eig_vecs = eig_vecs[:,idx]  
                return eig_vals, eig_vecs
        else:
            raise NotImplementedError('Covariance matrix is not symmetric')
        eig_values, eig_vecs = sorted_eig(cov)
        return eig_values, eig_vecs

    def _cov_plot(self):
        import seaborn as sns
        cov = self._covariance()
        cols =  self.train_x.columns
        sns.heatmap(cov, yticklabels = cols, xticklabels = cols,
                    annot = True, cbar = True, square = True
                    )

    def _var_plot(self, max_var):
        plt.plot(np.arange(0, self.train_x.shape[1]), max_var)
        plt.ylabel('Maximum variance'), plt.xlabel('Total number of features')
        plt.title('PCA : Maximum Variance vs Total number of features')
        plt.show()

    def _fit(self):
        eig_values, eig_vecs = self._eigen_vectors()
   
        max_var = [
            np.absolute(eig_values[idx]) * 100 / np.sum(np.absolute(eig_values))
            for idx in range(len(eig_values))
        ]

        self._var_plot(max_var)

        # Project the data onto principal component space
        eigen_pairs = np.concatenate((np.abs(eig_values).reshape(
            eig_values.size, 1), eig_vecs), axis = 1)
        eigen_pairs = np.array(sorted(eigen_pairs, key = lambda a_entry: a_entry[0], reverse = True))
        eig_vecs = eigen_pairs[:,1:]

        assert self.components > 0, "Number of components must be greater than 0"
        Eigen_vecs = np.delete(eig_vecs, range(self.components, eig_vecs.shape[1]), axis = 0)[::-1]
        return self.train_x.dot(Eigen_vecs.T)
