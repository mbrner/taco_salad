from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from taco_salat.toppings import ConfidenceCutter, criteria

import numpy as np


if __name__ == '__main__':
    X, y = make_classification(n_samples=100000)

    test_size = int(5e4)
    X_train = X[test_size:, 1:]
    y_train = y[test_size:]

    X_test = X[:test_size, 1:]
    y_test = y[:test_size]

    clf = RandomForestClassifier(n_estimators=12, n_jobs=3)

    clf.fit(X_train, y_train)

    conf = clf.predict_proba(X_test)[:, 1]

    pur_criteria = criteria.purity_criteria(threshold=0.99)
    conf_cutter = ConfidenceCutter(criteria)
    X = np.vstack((conf, X_test[:, 0]))
    conf_cutter.fit(X.T ,y_train)

    print(roc_auc_score(y_test, conf))
    plt.hist(conf)
    plt.show()
