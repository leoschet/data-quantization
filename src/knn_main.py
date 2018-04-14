from app.classifier.knn import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

import numpy as np

import codecs

def run_knn(data, lvq_id):
    results = {}
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    ks = [1, 2, 3]
    # ks = [5]

    verbose = 3 # Can be 0, 1, 2 or 3

    for relation in data:
        results[relation] = {}
        file_ = codecs.open('../res/results/' + relation + '_' + lvq_id + '.txt', 'w+', 'utf-8')

        print('\nTesting for ' + relation + ' data set')
        relation_data = data[relation]['data'] # numpy array
        relation_labels = data[relation]['labels'] # numpy array
        data_len = len(relation_data)
        print('\tTotal data collected: ' + str(data_len))

        features_types = data[relation]['features_types']

        results[relation] = {}
        results[relation]['precision'] = []
        results[relation]['recall'] = []
        results[relation]['f1'] = []

        results[relation]['precision-std'] = []
        results[relation]['recall-std'] = []
        results[relation]['f1-std'] = []
        for k in ks:

            knn = KNeighborsClassifier(k, weighted_distance=True)
            
            metrics = []
            train_data = relation_data
            train_labels = relation_labels
            test_data = data[relation]['test_data']
            test_labels = data[relation]['test_labels']
            
            if verbose > 0:
                print('\n\tTraining K-NN for new kfold configuration...')

            knn.fit(train_data, train_labels, features_types)

            if verbose > 0:
                print('\tClassifying data_test...')

            test_data_len = len(test_data)
            pred_labels = []
            for index, features in enumerate(test_data):
                ordered_pred_labels = knn.predict(features)
                pred_label, distances = ordered_pred_labels[0]
                pred_labels.append(pred_label)

            metrics.append(precision_recall_fscore_support(test_labels, pred_labels, average='weighted'))


            precisions = [precision for precision, _, _, _ in metrics]
            recalls = [recall for _, recall, _, _ in metrics]
            f1s = [f1 for _, _, f1, _ in metrics]

            results[relation]['precision'].append(np.mean(precisions))
            results[relation]['recall'].append(np.mean(recalls))
            results[relation]['f1'].append(np.mean(f1s))

            results[relation]['precision-std'].append(np.std(precisions))
            results[relation]['recall-std'].append(np.std(recalls))
            results[relation]['f1-std'].append(np.std(f1s))

            file_.write('Results using k = ' + str(knn.k))
            file_.write('\n\tAverage precision: ' + str(np.mean(precisions)))
            file_.write('\n\tPrecision standard deviation: ' + str(np.std(precisions)))
            file_.write('\n\tAverage recall: ' + str(np.mean(recalls)))
            file_.write('\n\tRecall standard deviation: ' + str(np.std(recalls)))
            file_.write('\n\tAverage F1-Score: ' + str(np.mean(f1s)))
            file_.write('\n\tF1-Score standard deviation: ' + str(np.std(f1s)))
            file_.write('\n\n=======================\n\n')

    return results, ks
