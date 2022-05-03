import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Visualise:

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = np.squeeze(train_y)

    def raw_data_trends(self):
        x = self.train_x[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']]
        total = pd.concat([x, self.train_y], axis = 1) 
        total[total['Classes'] == 0].plot(subplots = True, sharex = True, figsize = (10, 10), label = 'No Fire')
        plt.suptitle('Features vs No fire (0)', x = 0.5, y = 0.92, size = 15)
        plt.legend(loc = 'best')
        plt.show()
        total[total['Classes'] == 1].plot(subplots = True, sharex = True, figsize = (10, 10), label = 'No Fire')
        plt.suptitle('Features vs Fire (1)', x = 0.5, y = 0.92, size = 15)
        plt.legend(loc = 'best')
        plt.show()


    def data_plotting(self):
        plt.figure(figsize = (10, 10))
        c = 0
        for idx in self.train_x.columns:
            plt.subplot(5, 2, c + 1)
            sns.histplot(self.train_x[idx][self.train_y == 1], color='r',
                        label = 'Fire', kde = True, stat = 'density', linewidth = 0)
            sns.histplot(self.train_x[idx][self.train_y == 0], color = 'g',
                        label = 'No Fire', kde = True, stat = 'density', linewidth = 0)
            plt.legend()
            c += 1
        plt.suptitle('Histograms', size = 20)
        plt.subplots_adjust(top = 0.95)
        plt.tight_layout()
        plt.show()


    def scatter_plot(self):
        train_x = self.train_x[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']]
        plt.figure(figsize = (10, 10))
        c = 1
        for idx in train_x.columns:
            plt.subplot(3, 4, c)
            sns.scatterplot(data = train_x, x = train_x[idx], y = self.train_y)
            c += 1
        plt.suptitle('Scatter Plot', size = 20)
        plt.subplots_adjust(top = 0.95)
        plt.tight_layout()
        plt.show()