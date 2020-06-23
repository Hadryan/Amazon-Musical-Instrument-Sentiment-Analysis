from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score

import os, pickle


def run(data,  under_sampling=None):
    path = 'Data/pickles/rfr' if not under_sampling else 'Data/pickles/new_approach/rfr'

    if not os.path.isfile(path):
        params = {
            'n_estimators': [10, 50, 100, 300, 500, 700, 1000, 1500, 2000],
            'max_depth': [1, 3, 5],
            'bootstrap': [True, False],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'oob_score': [True, False]
        }

        grid = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=5, verbose=3, n_jobs=-1)

        grid.fit(data['x_train'], data['y_train'])

        with open(path, 'wb') as file:
            pickle.dump(grid, file)
    else:
        with open(path, 'rb') as file:
            grid = pickle.load(file)
    model = grid.best_estimator_

    y_pred = model.predict(data['x_test'])

    print(f'RFR: {accuracy_score(data["y_test"], y_pred)}')

    return {'acc': confusion_matrix(data['y_test'], y_pred), 'grid': grid}