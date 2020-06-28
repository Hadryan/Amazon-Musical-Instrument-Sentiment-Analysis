import pandas as pd
import Preprocessing

import pickle, os

from Models import NaiveBayes, RFR, XGBoost, LSTM


def main():
    df = pd.read_csv('Data/Amazon_Instrument_Reviews/Musical_instruments_reviews.csv')
    df_copy = df.copy()
    df_copy_2 = df.copy()
    df_copy_3 = df.copy()
    train_test_data = Preprocessing.run(df, under_sampling=False)

    # Holds confusion matrices for visualizations after testing
    results_dict = {}

    results_dict['mnb'] = NaiveBayes.run(train_test_data, 'Data/pickles/mb_grid')
    results_dict['rf'] = RFR.run(train_test_data, 'Data/pickles/rfr')
    results_dict['xgboost'] = XGBoost.run(train_test_data, 'Data/pickles/xgboost')
    results_dict['lstm'] = LSTM.run(train_test_data, 'Data/pickles/lstm', 'Data/pickles/lstm_weights.h5')

    with open('Data/pickles/results_dict_no_lemma', 'wb') as file:
        pickle.dump(results_dict, file)
    print(results_dict)

    train_test_data = Preprocessing.run(df_copy, under_sampling=None)

    results_dict = {}

    results_dict['mnb'] = NaiveBayes.run(train_test_data, 'Data/pickles/new_approach/mb_grid')
    results_dict['rf'] = RFR.run(train_test_data, 'Data/pickles/new_approach/rfr')
    results_dict['xgboost'] = XGBoost.run(train_test_data, 'Data/pickles/new_approach/xgboost')
    results_dict['lstm'] = LSTM.run(train_test_data, 'Data/pickles/new_approach/lstm', 'Data/pickles/new_approach/lstm_weights.h5')

    with open('Data/pickles/new_approach/results_dict_no_lemma', 'wb') as file:
         pickle.dump(results_dict, file)
    print(results_dict)

    train_test_data = Preprocessing.run(df_copy_2, under_sampling=True)

    results_dict = {}

    results_dict['mnb'] = NaiveBayes.run(train_test_data, 'Data/pickles/new_approach_2/mb_grid')
    results_dict['rf'] = RFR.run(train_test_data, 'Data/pickles/new_approach_2/rfr')
    results_dict['xgboost'] = XGBoost.run(train_test_data, 'Data/pickles/new_approach_2/xgboost')
    results_dict['lstm'] = LSTM.run(train_test_data, 'Data/pickles/new_approach_2/lstm', 'Data/pickles/new_approach_2/lstm_weights.h5')

    with open('Data/pickles/new_approach_2/results_dict_no_lemma', 'wb') as file:
         pickle.dump(results_dict, file)
    print(results_dict)

    train_test_data = Preprocessing.run(df_copy_3, under_sampling=True, lemma=True)

    results_dict = {}

    results_dict['mnb'] = NaiveBayes.run(train_test_data, 'Data/pickles/lemma/mb_grid')
    results_dict['rf'] = RFR.run(train_test_data, 'Data/pickles/lemma/rfr')
    results_dict['xgboost'] = XGBoost.run(train_test_data, 'Data/pickles/lemma/xgboost')
    results_dict['lstm'] = LSTM.run(train_test_data, 'Data/pickles/lemma/lstm',
                                    'Data/pickles/lemma/lstm_weights.h5')

    with open('Data/pickles/lemma/results_dict_lemma', 'wb') as file:
        pickle.dump(results_dict, file)
    print(results_dict)


if __name__ == '__main__':
    main()
