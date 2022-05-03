import numpy as np

def confusion_matrix(tn, tp, fn, fp):
    confusion_matrix = np.array([[tn, fp],[fn, tp]])
    return confusion_matrix

def calculate_f1_score(tn, tp, fn, fp):
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)
    return f1_score

def calculate_performance(predicted_label_arr, labels_validation):
    tn , fp, fn, tp = 0, 0, 0, 0
    for i in range(labels_validation.shape[0]):
        if(labels_validation[i] == 0):
            if(labels_validation[i] == predicted_label_arr[i]):
                tn += 1
            else:
                fp += 1
        else:
            if(labels_validation[i] == predicted_label_arr[i]):
                tp += 1
            else:
                fn += 1
    f1_score = calculate_f1_score(tn, tp, fn, fp)
    conf_matrix = confusion_matrix(tn, tp, fn, fp)
    return conf_matrix, f1_score


