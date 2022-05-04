import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from src import *
from utils import *
from configs import *
from models import *


def time_series(col):
    days = [
        col['Date'][row].day
        for row in range(len(col))
    ]
    months = [
        col['Date'][row].month
        for row in range(len(col))
    ]
    return days, months


def dataloader(args):
    train_path = os.path.join(args.root, 'data/' + args.train_file)
    test_path = os.path.join(args.root, 'data/' + args.test_file)
    dataset = Dataset(args)

    if args.train_mode == 'whole':
        print('--- Loading the original dataset ---')
        train_x, train_y = dataset.load_data(train_path)
        test_x, test_y = dataset.load_data(test_path)

        if args.if_time:
            print('--- Time series processing ---')
            train_days, _ = time_series(train_x)[0], time_series(train_x)[1]
            test_days, _ = time_series(test_x)[0], time_series(test_x)[1]
            train_x['Day_sin'] = pd.Series(dataset.day_month_transform(train_days)[0])
            train_x['Day_cos'] = pd.Series(dataset.day_month_transform(train_days)[1])
            test_x['Day_sin'] = pd.Series(dataset.day_month_transform(test_days)[0])
            test_x['Day_cos'] = pd.Series(dataset.day_month_transform(test_days)[1])

        return train_x, train_y, test_x, test_y
    else:
        train_x, train_y, test_x, test_y = dataset.run()
        return train_x, train_y, test_x, test_y


def load_models(args, train_x, train_y, test_x, test_y):
    if args.model == 'base':
        model = NearestMeans(train_x.to_numpy(), train_y)
        pred = model.runner(test_x.to_numpy(), test_y)
        return accuracy_score(test_y, pred), pred
    elif args.model == 'trivial':
        model = Trivial(train_x.to_numpy(), train_y)
        preds = model.fit(test_y)
        return accuracy_score(test_y, preds), preds
    elif args.model == 'perceptron':
        model = Perceptron()
        _, weights = model._fit(train_x.to_numpy(), train_y)
        preds = model._classify(test_x.to_numpy(), weights)
        return accuracy_score(test_y, preds), preds
    elif args.model == 'knn':
        model = KNN(
            train_x.to_numpy(), 
            train_y,
            test_x.to_numpy(),
            test_y,
            args.k
            )
        preds = model.fit()
        return accuracy_score(test_y, preds), preds
    elif args.model == 'svm':
        model = svm.SVC(
            kernel = args.svm_kernel,
            gamma = 'auto',
            random_state = 2, 
            C = args.svm_c
        )
        model.fit(train_x.to_numpy(), train_y)
        preds = model.predict(test_x.to_numpy())
        return accuracy_score(test_y, preds), preds
    elif args.model == 'dt':
        print('--- Running the Decision Tree model ---')
        model = DecisionTreeClassifier(
            criterion = 'gini', 
            random_state = 2,
            max_depth = args.dt_max_depth
            )        
        model.fit(train_x.to_numpy(), train_y)
        preds = model.predict(test_x.to_numpy())
        return accuracy_score(test_y, preds), preds
    elif args.model == 'rf':
        model = RandomForestClassifier(
            n_estimators = args.n_estimators,
            criterion = 'gini',
            max_depth = args.rf_max_depth, 
            random_state = 2 
        )
        model.fit(train_x.to_numpy(), train_y)
        preds = model.predict(test_x.to_numpy())
        return accuracy_score(test_y, preds), preds
    elif args.model == 'naive':
        model = GaussianNB(var_smoothing = args.var_smoothing)
        model.fit(train_x.to_numpy(), train_y)
        preds = model.predict(test_x.to_numpy())
        return accuracy_score(test_y, preds), preds
    else:
        SystemError('Model not found')