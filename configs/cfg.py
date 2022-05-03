import argparse

def get_args():
    """
    Define all the configurations of the project here.
    Run the program from the terminal by "--" the args.
    
    Returns
    --------
         a list of parsed arguments. 
    """ 
    parser = argparse.ArgumentParser(description='EE_559 Project')
    parser.add_argument('--seed', type = int,
                       default = 0,
                       help = 're-produce the results with random seed') 
    parser.add_argument('--data_viz', action = 'store_true',
                        help = 'visualise the original data')
    
    '''
    Params for I/O locations
    '''
    parser.add_argument('--root', type = str,
                        default = '/home/sarthak/Desktop/spring_22/EE_559/project/',
                        help = 'root directory of the project')
    parser.add_argument('--train_file', type = str,
                        default = 'algerian_fires_train.csv',
                        help = 'train file')
    parser.add_argument('--test_file', type = str,
                        default = 'algerian_fires_test.csv',
                        help = 'test file')
    parser.add_argument('--kfold', type = int,
                         default = 5,
                         help = 'number of folds for cross-validation setup.')
    parser.add_argument('--kfold_loc', type = str,
                         default = '/home/sarthak/Desktop/spring_22/EE_559/project/data_k/',     
                         help = 'location of the kfold file')

    '''
    Params for pre-processing.
    '''
    parser.add_argument('--past_days', type = int,
                         default = 2,
                         help = 'past days to be considered for feature engineering')

    '''
    Data mode
    '''
    parser.add_argument('--train_mode', type = str,
                        default = 'stats',
                        help = 'train mode, available: whole, stats')
    parser.add_argument('--if_time', action = 'store_true',
                        help = 'if temporal data is to be considered.')

    '''
    Params for model hyperparameters
    '''
    parser.add_argument('--model', type = str,
                         default = 'base',
                         help = 'model to be used, available: mlp, svm, rf, naive, knn, base, trivial, dt, rf')
    parser.add_argument('--k', type = int, default = 5,
                         help = 'number of neighbors for KNN')
    args = parser.parse_args()

    return args
