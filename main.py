#!/usr/bin/env python
# coding: utf-8

##################################################
# Author:   Sarthak Kumar Maharana, Shoumik Nandi
# Email:    maharana@usc.edu, shoumikn@usc.edu
# Date:     05/03/2022
# Course:   EE 559
# Objective:  Final Project
# Instructor: Prof. B Keith Jenkins
##################################################

import os
import random
import shutil
import numpy as np
import pandas as pd

from src import *
from utils import *
from configs import *
from models import *
from metrics import *
from runner import *

from sklearn.preprocessing import MinMaxScaler

def main(args):
    
    if args is None:
        SystemExit('No arguments passed!!')
        
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    train_x, train_y, test_x, test_y = dataloader(args)

    print('--- Data Visualization ---')
    if args.data_viz and args.train_mode == 'whole':
        viz = Visualise(train_x, train_y)
        viz.raw_data_trends()
        viz.data_plotting()
        viz.box_plot()
        viz.scatter_plot()
    
    train_x = train_x.drop(['Date', 'BUI'], axis = 1)
    test_x = test_x.drop(['Date', 'BUI'], axis = 1)

    train_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train_x), index = train_x.index, columns = train_x.columns)
    test_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test_x), index = test_x.index, columns = test_x.columns)

    print('--- Selecting the best features ---')
    fs = BestFeatures(train_scaled, train_y)
    fs._runner()

    if args.train_mode == 'whole':
        train_scaled = train_scaled
        test_scaled = test_scaled
    else:
        if not args.if_time:
            train_scaled = train_scaled[[
                'ISI', 'DC', 'DMC', 'FFMC', f'Temperature-max-{args.past_days}', f'Rain-max-{args.past_days}']]
            test_scaled = test_scaled[['ISI', 'DC', 'DMC',  'FFMC',
                                       f'Temperature-max-{args.past_days}', f'Rain-max-{args.past_days}']]
        else:
            train_scaled = train_scaled[['ISI', 'DC', 'DMC', 'FFMC', 'Day_cos',
                                         f'Temperature-max-{args.past_days}', f'Rain-max-{args.past_days}']]
            test_scaled = test_scaled[['ISI', 'DC', 'DMC', 'FFMC', 'Day_cos',
                                       f'Temperature-max-{args.past_days}', f'Rain-max-{args.past_days}']]


    print(f"--- Performing 5-fold cross validation ---")
    os.makedirs(args.kfold_loc)
    k_fold_split(train_scaled, train_y, args)

    print(f"--- Model selection: {args.model} ---")
    average_acc = []
    average_f1 = []

    for fold_idx in range(int(args.kfold)):
        train_x_k = pd.read_csv(os.path.join(args.kfold_loc, f'{fold_idx + 1}_x_train.csv'))
        train_y_k = pd.read_csv(os.path.join(args.kfold_loc, f'{fold_idx + 1}_y_train.csv'))
        val_x_k = pd.read_csv(os.path.join(args.kfold_loc, f'{fold_idx + 1}_x_val.csv'))
        val_y_k = pd.read_csv(os.path.join(args.kfold_loc, f'{fold_idx + 1}_y_val.csv'))

        train_y_k, val_y_k = np.squeeze(
            train_y_k.to_numpy()), np.squeeze(val_y_k.to_numpy())

        acc, preds = load_models(
                args, train_x_k, train_y_k, val_x_k, val_y_k
            )
        average_acc.append(acc)
        average_f1.append(calculate_performance(val_y_k, preds)[1])

    print(f"Avg validation acc: {np.mean(np.array(average_acc))}")
    print(f"Avg validation f1: {np.mean(np.array(average_f1))}")


    if os.path.exists(args.kfold_loc):
        shutil.rmtree(args.kfold_loc)


    print(f"--- Test performance ---")
    acc, test_preds = load_models(args, train_scaled, train_y, test_scaled, test_y)
    cm, f1 = calculate_performance(test_preds, test_y)
    print(f"Test accuracy: {acc}")    
    print(f"Test F1 score: {f1}")
    print(f"Test confusion matrix: \n{cm}")



if __name__ == '__main__':
    PATH = os.getcwd()
    args = get_args()
    main(args)


