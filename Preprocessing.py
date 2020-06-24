import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# nltk.download('punkt')


def run(df, under_sampling=None):

    stop_words = set(stopwords.words('english'))

    df.drop(['reviewerID', 'reviewerName', 'unixReviewTime', 'reviewTime', 'asin', 'helpful'], axis=1, inplace=True)

    df.dropna(inplace=True)

    df['full_text'] = df['summary'] + " " + df['reviewText']

    df.drop(['summary', 'reviewText'], axis=1, inplace=True)

    clean_text = []

    if not under_sampling:

        df['overall'] = df['overall'].replace({1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'})

        # NLP stuff

        df['full_text'] = df['full_text'].apply(lambda x: x.lower())
        for text in df['full_text']:
            words = word_tokenize(text)
            words = [word for word in words if word.isalpha()]
            words = [word for word in words if word not in stop_words]
            clean_text.append(' '.join(words))

        df['clean_text'] = clean_text
        x = df['clean_text']
        y = df['overall']
        x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True)

        vectorizer = TfidfVectorizer(max_features=8000)

        x_train = vectorizer.fit_transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()

        y_train = y_train.replace({'positive': 2, 'neutral': 1, 'negative': 0})
        y_test = y_test.replace({'positive': 2, 'neutral': 1, 'negative': 0})

    elif under_sampling is None:
        df['overall'] = df['overall'].replace(
            {1: 'negative', 2: 'negative', 3: 'negative', 4: 'positive', 5: 'positive'})

        # NLP stuff
        df['full_text'] = df['full_text'].apply(lambda x: x.lower())
        for text in df['full_text']:
            words = word_tokenize(text)
            words = [word for word in words if word.isalpha()]
            words = [word for word in words if word not in stop_words]
            clean_text.append(' '.join(words))

        df['clean_text'] = clean_text
        x = df['clean_text']
        y = df['overall']
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True)

        vectorizer = TfidfVectorizer(max_features=8000)

        x_train = vectorizer.fit_transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()

        y_train = y_train.replace({'positive': 1, 'negative': 0})
        y_test = y_test.replace({'positive': 1, 'negative': 0})

    else:

        df['overall'] = df['overall'].replace(
            {1: 'negative', 2: 'negative', 3: 'negative', 4: 'positive', 5: 'positive'})

        ros = RandomUnderSampler()

        new_x, new_y = ros.fit_resample(np.reshape(df['full_text'].values, (-1,1)), df['overall'])

        new_x = [text.lower() for x in new_x for text in x]

        for text in new_x:
            words = word_tokenize(text)
            words = [word for word in words if word.isalpha()]
            words = [word for word in words if word not in stop_words]
            clean_text.append(' '.join(words))

        x_train, x_test, y_train, y_test = train_test_split(new_x, new_y, shuffle=True)

        vectorizer = TfidfVectorizer(max_features=8000)

        x_train = vectorizer.fit_transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()

        y_train = y_train.replace({'positive': 1, 'negative': 0})
        y_test = y_test.replace({'positive': 1, 'negative': 0})

    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

