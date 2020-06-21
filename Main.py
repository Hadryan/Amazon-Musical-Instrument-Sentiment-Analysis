import pandas as pd
import Preprocessing, Visualize

import pickle

from Models import NaiveBayes, RFR, XGBoost, LSTM


def main():
    df = pd.read_csv('Data/Amazon_Instrument_Reviews/Musical_instruments_reviews.csv')

    # Visualize.run(df)

    train_test_data = Preprocessing.run(df)

    # Holds confusion matrices for visualizations after testing
    results_dict = {}

    results_dict['mnb'] = NaiveBayes.run(train_test_data)
    results_dict['rfr'] = RFR.run(train_test_data)
    results_dict['xgboost'] = XGBoost.run(train_test_data)
    #results_dict['lstm'] = LSTM.run(train_test_data)
    
    with open('Data/pickles/results_dict_no_lemma', 'wb') as file:
        pickle.dump(results_dict, file)
    print(results_dict)

if __name__ == '__main__':
    main()
