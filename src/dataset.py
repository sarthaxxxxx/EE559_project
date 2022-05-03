import os
import math
import numpy as np   
import pandas as pd


class Dataset:
    def __init__(self, args):
        self.train_path = os.path.join(args.root,
                                       'data/' + args.train_file)
        self.test_path = os.path.join(args.root,
                                      'data/' + args.test_file)
        self.past_days = args.past_days 
        self.if_time = args.if_time
 
    @staticmethod
    def date_conversion(df, flag = False):
        if flag:
            df['Date'] = df['Date'].iloc[0 : len(df)].dt.strftime('%d/%m/%Y')
            return df
        df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
        return df

    def load_data(self, path):
        df = pd.read_csv(path)
        x, y = df.iloc[:, :-1],\
               df.iloc[:, -1]
        return self.date_conversion(x), y

    @staticmethod
    def day_month_transform(col):
        sine = [math.sin((2 * np.pi * vals) / max(col)) for vals in list(col)]
        cosine = [math.cos((2 * np.pi * vals) / max(col)) for vals in list(col)]
        return sine, cosine
    
    def feature_engg(self):

        train_start_date = self.train_x['Date'][0]
        test_start_date = self.test_x['Date'][0]

        total_data = pd.concat([self.train_x, self.test_x]).reset_index(drop = True)

        # Rolling window (mean, max, min, median) 
        for feats in ['Temperature' ,'RH' ,'Ws', 'Rain']:
            for mode in ['mean', 'max', 'min', 'median']:
                print(f"Calculating {mode} of {feats} for past {self.past_days} days!")
                mask_end = train_start_date + pd.Timedelta(days = self.past_days - 1)
                init_subset = total_data[
                    (total_data['Date'] >= train_start_date) &
                    (total_data['Date'] <= mask_end)
                ]

                temp = init_subset[str(feats)].tolist()
                stats = [0.0] * 2
                row = 2
                while row < len(temp):
                    stats.extend([eval(str('np.') + str(mode))(temp[:row])] * 2)
                    row += 2
                
                idx = 0
                jdx = (total_data['Date'].tolist()).index(
                    mask_end + pd.Timedelta(days = 1))
                test_start_idx = (total_data['Date'].tolist()).index(test_start_date)
                while jdx <= len(total_data) - 1:
                    if jdx == test_start_idx: train_end_idx = idx
                    data_subset = total_data.loc[idx: jdx - 1].drop(columns = ['Date'])
                    stats.extend([eval(
                    str('np.') + str(mode))(data_subset[str(feats)])] * 2)
                    idx += 2
                    jdx += 2
                total_data = pd.concat([total_data, pd.Series(stats).rename(
                    f'{feats}-{mode}-{self.past_days}')], axis = 1
                    )


        if self.if_time:
            print('--- Time series processing ---')
            days, months = [], []
            for row in range(len(total_data)):
                days.append(total_data['Date'][row].day)
                months.append(total_data['Date'][row].month)

            total_data['Day_sin'] = pd.Series(self.day_month_transform(days)[0])
            total_data['Day_cos'] = pd.Series(self.day_month_transform(days)[1])
            total_data['Month_sin'] = pd.Series(self.day_month_transform(months)[0])
            total_data['Month_cos'] = pd.Series(self.day_month_transform(months)[1])

        return total_data, train_end_idx, test_start_idx

    def split_data(self, total_data, train_end, test_start):
        train_x, train_y = total_data.loc[: train_end - 1], self.train_y[: train_end]
        test_x = total_data.loc[test_start: ]
        return train_x, train_y, test_x, self.test_y
    
    def run(self):
        self.train_x, self.train_y = self.load_data(self.train_path)
        self.test_x, self.test_y = self.load_data(self.test_path)
        total_data, train_end_idx, test_start_idx = self.feature_engg()
        train_x, train_y, test_x, test_y = self.split_data(
            total_data, train_end_idx, test_start_idx
            )
        return self.date_conversion(train_x, True), train_y, \
        self.date_conversion(test_x, True), test_y

