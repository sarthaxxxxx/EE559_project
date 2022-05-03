import pandas as pd

def k_fold_split(x, y, args, folds = 5):
    assert len(x) == len(y), 'train and labels must have the same length'
  
    q = int(len(x) / folds)
    total_n = int(q * folds)
    
    train_x = x[:total_n]
    train_y = y[:total_n]   

    splits = [
        [split_idx, split_idx + int(len(train_x) / folds)]
        for split_idx in range(0, len(train_x), int(len(train_x) / folds))
    ]

    days_to_remove = 0 if args.train_mode == 'whole' else 2 * args.past_days
    
    PATH = str(args.kfold_loc)
    
    for fold_idx in range(len(splits)):
        val_x = train_x[splits[fold_idx][0]:splits[fold_idx][1]]
        val_y = train_y[splits[fold_idx][0]:splits[fold_idx][1]]
        if fold_idx == 0:
            subset_x = train_x[splits[fold_idx][1] + days_to_remove:]
            subset_y = train_y[splits[fold_idx][1] + days_to_remove:]
        elif fold_idx in [1, 2, 3]:
            subset_x = pd.concat([train_x[:splits[fold_idx][0] - days_to_remove],
                                  train_x[splits[fold_idx][1] + days_to_remove:]], axis = 0)
            subset_y = pd.concat([train_y[:splits[fold_idx][0] - days_to_remove],
                                  train_y[splits[fold_idx][1] + days_to_remove:]], axis = 0)
        elif fold_idx == 4:
            subset_x = train_x[:splits[fold_idx][0] - days_to_remove]
            subset_y = train_y[:splits[fold_idx][0] - days_to_remove]
        else:
            raise ValueError(f"Fold index {fold_idx} is not supported")

        subset_x.to_csv(str(PATH +
                             str(fold_idx + 1) + '_x_train.csv'), index = False)
        subset_y.to_csv(str(PATH +
                             str(fold_idx + 1) + '_y_train.csv'), index = False)
        val_x.to_csv(str(PATH + str(fold_idx + 1) +
                            '_x_val.csv'), index = False)
        val_y.to_csv(str(PATH + str(fold_idx + 1) +
                            '_y_val.csv'), index = False)
