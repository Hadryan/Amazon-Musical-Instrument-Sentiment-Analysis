from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score

import os, pickle


def run(data, under_sampling=None):

    path = 'Data/pickles/lstm' if not under_sampling else 'Data/pickles/new_approach/lstm'

    if not os.path.isfile(path):
        '''params = {
            'n_estimators': [10, 20, 30, 40, 50, 100, 250, 500, 1000],
            'max_depth': [1, 3, 5],
            'learning_rate': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.3, 0.5, 0.7, 1],
            'reg_alpha': [0, 0.001, 0.1, 0.5, 1, 2, 5],
            'reg_lambda': [0, 0.001, 0.1, 1, 2, 5]
        }'''

        params = {
            'n_estimators': [100],
            'max_depth': [3],
            'learning_rate': [0.04],
            'reg_alpha': [0, 0.001, 0.1, 0.5, 1, 2, 5],
            'reg_lambda': [0, 0.001, 0.1, 1, 2, 5]
        }

        grid = GridSearchCV(XGBClassifier(tree_method='gpu_hist'), param_grid=params, cv=5, verbose=3)

        grid.fit(data['x_train'], data['y_train'])

        with open(path, 'wb') as file:
            pickle.dump(grid, file)
    else:
        with open(path, 'rb') as file:
            grid = pickle.load(file)
    model = grid.best_estimator_

    y_pred = model.predict(data['x_test'])

    print(f'xgboost: {accuracy_score(data["y_test"], y_pred)}')

    return {'acc': confusion_matrix(data['y_test'], y_pred), 'grid': grid}