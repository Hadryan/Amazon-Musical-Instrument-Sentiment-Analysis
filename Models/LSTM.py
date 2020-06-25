from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential, model_from_json

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pickle, os
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(data, path, weights_path):

    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    print(y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model = Sequential()

    model.add(Embedding(5000, 32, input_length=x_train.shape[1]))

    model.add(
        LSTM(128, return_sequences=True)
    )

    model.add(Dropout(0.2))

    model.add(
        LSTM(128)
    )

    model.add(Dropout(0.2))

    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    model.compile(loss='sparse_categorical_cross'
                       'entropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    if not os.path.isfile(path):

        history = model.fit(x_train, y_train, epochs=200, verbose=1, validation_data=(x_val, y_val),
                            callbacks=callback)

        model.save_weights(weights_path)

        model_stuff = {'hist': history.history}

        with open(path, 'wb') as file:
            pickle.dump(model_stuff, file)

    else:
        with open(path, 'rb') as file:
            model_stuff = pickle.load(file)

            model.load_weights(weights_path)

            model.compile(loss='sparse_categorical_cross'
                               'entropy', optimizer=opt, metrics=['accuracy'])

    y_pred = model.predict(x_test)

    y_pred = interpret_predictions(y_pred.tolist())

    return {'acc': confusion_matrix(y_test, y_pred), 'hist': model_stuff['hist']}


def interpret_predictions(y_pred):

    result = []

    for i in y_pred:

        order = {}

        for j in range(len(i)):
            order[j] = i[j]

        result.append(max(order, key=order.get))
    return result