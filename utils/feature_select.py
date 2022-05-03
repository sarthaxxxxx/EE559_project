import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .PCA import PCA


class BestFeatures:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = np.squeeze(train_y)
    
    def _check(self):
        assert type(self.train_x) == pd.DataFrame, 'train_x must be a pandas DataFrame'
        assert type(self.train_y) == pd.Series, 'train_y must be a pandas Series'
        assert len(self.train_x) == len(self.train_y), 'train_x and train_y must have same number of data points'
        return True

    def pcc(self):
        if self._check():
            # normalising won't change the result. PCC is independent of change in origin and scale. 
            def corr(x1, x2):
                cc = ((x1 * x2).mean() - (x1.mean() * x2.mean())) / \
                    np.sqrt((x1.var() * x2.var()))
                return cc
            cc = [corr(self.train_x.to_numpy()[:,idx], self.train_y.to_numpy())
                  for idx in range(self.train_x.shape[1])]
            # total_data = pd.concat([self.train_x, self.train_y], axis = 1)
            return pd.Series(cc, index = self.train_x.columns).sort_values(ascending = False)

    def pca(self):
        if self._check():
            pca = PCA(self.train_x, components = 10)
            return pca._fit()
    
    def sfs(self):
        model = svm.SVC(kernel = 'linear')
        sfs = SequentialFeatureSelector(model,
                                        n_features_to_select= 7,
                                        direction = 'forward')
        sfs.fit(self.train_x.to_numpy(), self.train_y.to_numpy())
        x = sfs.transform(self.train_x.to_numpy())
        print(f"Important features : {list(self.train_x.columns[sfs.get_support()])}")
        return x
        

    def covariance_plot(self):
        mean = np.mean(self.train_x.to_numpy(), axis = 0)
        std = np.std(self.train_x.to_numpy(), axis = 0)
        train_norm = (self.train_x - mean) / std
        covariance = np.cov(train_norm.T)
        sns.heatmap(covariance, yticklabels = self.train_x.columns, xticklabels = self.train_x.columns,
                        annot = True, cbar = True, square = True
                    )
        plt.title('Covariance matrix of features.', size = 20)
        plt.tight_layout()
        plt.show()


    def rfe(self):
        model = svm.SVC(kernel = 'linear')
        rfe = RFE(estimator = model, n_features_to_select = 7)
        rfe.fit(self.train_x.to_numpy(), self.train_y.to_numpy())
        plt.barh(self.train_x.columns, rfe.ranking_)
        plt.xlabel('Feature Importance'), plt.ylabel('Features'), plt.title('RFE')

    def mutual_info(self):
        scores = mutual_info_classif(self.train_x.to_numpy(), self.train_y.to_numpy())
        for idx in sorted(zip(scores, self.train_x.columns), reverse = True):
            print(f"{idx[1]} : {idx[0]}")
        
    def anova(self):
        if self._check():
            fs = SelectKBest(f_classif, k = 7)
            fs.fit_transform(self.train_x.to_numpy(), self.train_y.to_numpy())
            plt.barh(self.train_x.columns, fs.get_support())
            plt.xlabel('F-score'), plt.ylabel('Features'), plt.title('ANOVA')

    def rf(self):
        if self._check():
            model = RandomForestClassifier(n_estimators = 10, criterion = 'gini')
            model.fit(self.train_x.to_numpy(), self.train_y.to_numpy())
            indices = np.argsort(model.feature_importances_)
            largest_indices = indices[::-1][:len(indices)]
            for i in largest_indices:
                print(self.train_x.columns[i],":", model.feature_importances_[i])
            plt.barh(self.train_x.columns, model.feature_importances_)
            plt.xlabel('Feature Importance'), plt.ylabel('Features'), plt.title('Random Forest')

    def dt(self):
        if self._check():
            model = DecisionTreeClassifier(criterion = 'gini')
            model.fit(self.train_x.to_numpy(), self.train_y.to_numpy())
            indices = np.argsort(model.feature_importances_)
            largest_indices = indices[::-1][:len(indices)]
            for i in largest_indices:
                print(self.train_x.columns[i],":", model.feature_importances_[i])
            plt.barh(self.train_x.columns, model.feature_importances_)
            plt.xlabel('Feature Importance'), plt.ylabel('Features'), plt.title('Decision Tree')

    def _runner(self):
        print('Pearson Correlation Coefficients between features and target !')
        print(self.pcc())
        print('PCA !')
        print(self.pca())
        print('Visualising the covariance matrix of features !')
        print(self.covariance_plot())
        print('Running aN SVM model using RFE and reporting accuracy on each feature !')
        print(self.rfe())
        print('Running an SVM model using SFS and reporting accuracy on each feature !')
        print(self.sfs())
        print('Calculating mutual information between features and target !')
        print(self.mutual_info())
        print('Running ANOVA on features and reporting the best features !')
        print(self.anova())
        print('Running a Random Forest model and reporting the best features !')
        print(self.rf())
        print('Running a Decision Tree model and reporting the best features !')
        print(self.dt())


    