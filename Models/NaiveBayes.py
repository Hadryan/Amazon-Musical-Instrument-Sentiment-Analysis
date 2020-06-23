from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score

import os, pickle


def run(data, under_sampling=None):

    path = 'Data/pickles/mb_grid' if not under_sampling else 'Data/pickles/new_approach/mb_grid'
    if not os.path.isfile(path):

        params = {
            'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10]
        }

        grid = GridSearchCV(MultinomialNB(), param_grid=params, cv=5, verbose=3)

        grid.fit(data['x_train'], data['y_train'])

        with open(path, 'wb') as file:
            pickle.dump(grid, file)

    else:
        with open(path, 'rb') as file:
            grid = pickle.load(file)

    model = grid.best_estimator_

    y_pred = model.predict(data['x_test'])

    print(accuracy_score(data['y_test'], y_pred))

    return {'acc': confusion_matrix(data['y_test'], y_pred), 'grid': grid}