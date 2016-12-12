import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

from taco_salad.toppings import ConfidenceCutter, criteria

if __name__ == '__main__':
    X, y = make_classification(n_samples=100000)

    test_size = int(5e4)
    X_train = X[test_size:, 1:]
    y_train = y[test_size:]

    X_test = X[:test_size, 1:]
    y_test = y[:test_size]

    clf = RandomForestClassifier(n_estimators=120, n_jobs=3)

    clf.fit(X_train, y_train)

    conf = clf.predict_proba(X_test)[:, 1]

    pur_criteria = criteria.purity_criteria(threshold=0.90)
    conf_cutter = ConfidenceCutter(positions=np.linspace(-2.2, 2.2, 100),
                                   window_size=0.8,
                                   n_bootstraps=100,
                                   n_jobs=3,
                                   criteria=pur_criteria)
    X = np.vstack((conf, X_test[:, 0]))

    conf_cutter.fit(X.T, y_train)
    x_curves = np.linspace(-2.2, 2.2, 1000)
    y_cut_curve = conf_cutter.cut_opts.cut_curve(x_curves)

    plt.hexbin(conf, X_test[:, 0], gridsize=50, cmap=plt.cm.plasma)
    plt.plot(y_cut_curve, x_curves, '--', color='w', lw=5,
             label='Confidence Cutter')
    plt.xlabel('Classifier Score')
    plt.ylabel('Observable')
    plt.legend(loc='best', title='Purity 90%')
    plt.xlim([0., 1.])
    plt.ylim([-2.2, 2.2])
    plt.show()
