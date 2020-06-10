
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential

import tensorflow as tf
import pickle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def run(data):

    model = Sequential()
    print(data['x_train'])
    y_train = data['y_train']
    y_test = data['y_test']

    y_train = y_train.replace({'positive': 2, 'neutral':1, 'negative': 0})
    y_test = y_test.replace({'positive': 2, 'neutral': 1, 'negative': 0})
    model.add(Embedding(5000, 32, input_length=data['x_train'].shape[1]))

    model.add(
        LSTM(128, return_sequences=True)
    )

    model.add(Dropout(0.2))

    model.add(
        LSTM(128)
    )

    model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model.compile(loss='sparse_categorical_cross'
                       'entropy', optimizer=opt, metrics=['accuracy'])
    
    print(model.summary())
    
    history = model.fit(data['x_train'], y_train, epochs=200, verbose=1, validation_data=(data['x_test'], y_test), callbacks=callback)

    with open('Data/pickles/lstm', 'wb') as file:
        pickle.dump(history, file)