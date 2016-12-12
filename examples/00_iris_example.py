import logging

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier


from taco_salad import TacoSalad

if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s|%(name)s|%(levelname)s| ' +
                        '%(message)s'), level=logging.INFO)

    logging.info('Staring TacoSalad Iris Sample')
    iris = load_iris()
    target = np.array(iris['target'], dtype=int)
    print(np.bincount(target))

    df = pd.DataFrame(data=iris['data'],
                      columns=iris['feature_names'])
    df['target'] = target
    logging.info('Loaded Iris Dataset.')

    salat = TacoSalad(df=df, roles=[0, 0, 0, 0, 1])

    salat.add_layer('clf1')
    salat.add_component('clf1',
                        'Dummy1',
                        clf=DummyClassifier(),
                        attributes=['layer0:attribute:*'],
                        label='target',
                        returns=3)

    salat.add_component('clf1',
                        'Dummy2',
                        clf=DummyClassifier(),
                        attributes=['layer0:attribute:*'],
                        label='target',
                        returns=3)

    salat.add_component('clf1',
                        'Dummy3',
                        clf=DummyClassifier(),
                        attributes=['layer0:attribute:*'],
                        label='target',
                        returns=3)

    salat.add_layer('clf2')
    salat.add_component('clf2',
                        'Dummy4',
                        clf=DummyClassifier(),
                        attributes=['clf1:*'],
                        label='target',
                        returns=3)

    salat.add_component('clf2',
                        'Dummy5',
                        clf=DummyClassifier(),
                        attributes=['layer0:attribute:*', 'clf1:Dumm1*'],
                        label='target',
                        returns=3)

    salat.add_component('clf2',
                        'Dummy6',
                        clf=DummyClassifier(),
                        attributes=['layer0:attribute:*', 'clf1:Dumm2*'],
                        label='target',
                        returns=3)

    salat.add_layer('clf3', n_jobs=2, predict_parallel=True)
    salat.add_component('clf3',
                        'DummyFinal',
                        clf=DummyClassifier(),
                        attributes=['*:Dummy*'],
                        label='target',
                        returns=['score_final_1',
                                 'score_final_2',
                                 'score_final3'])

    df = salat.fit_df(df, clear_df=False)
    df = pd.DataFrame(data=iris['data'],
                      columns=iris['feature_names'])
    df['target'] = target
    score = salat.predict_df(df)
