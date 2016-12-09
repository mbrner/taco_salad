from taco_salat import TacoSalat

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold
    iris = load_iris()
    target = np.array(iris['target'], dtype=int)

    df = pd.DataFrame(data=iris['data'],
                      columns=iris['feature_names'])
    df['target'] = target

    salat = TacoSalat(df, roles=[0, 0, 0, 0, 1])
    first_layer = salat.add_layer('FirstClassificationLayer')
    salat.add_component(first_layer,
                        name='RandomForest',
                        clf=RandomForestClassifier(),
                        attributes=['layer0:attribute:*'],
                        label='layer0:label:*',
                        returns=['score_1', 'score_2', 'score_3'],
                        weight=None,
                        roles=[0, 0, 0],
                        predict_func='predict_proba')
    kf = KFold(n_splits=3, shuffle=True)
    for train, test in kf.split(np.empty(df.shape)):
        df_train = df.loc[train, :]
        df_test = df.loc[test, :]
        salat.fit_df(df_train)
        salat.predict_df(df_test)
