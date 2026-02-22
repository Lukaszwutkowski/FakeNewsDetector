"""
Modul odpowiada za wyciaganie dodatkowych cech z tekstu. Glowna cecha to analiza sentymentu (VADER)
VADER (Valence Aware Dictionary and Sentiment Reasoner)

Dzialanie:

Predykcja =
-> uzytkownik wkleja tytul i tekst ->
-> predictor.py wywoluje extract_all_features - alias dl aget_sentiment() ->
-> analiza jednego tekstu i zwrot slownika z sentymentem

Trening =
-> model trenuje na calym datasecie ->
-> extract_features_batch(texts) przetwarza tysiace tekstow naraz ->
-> FeatureExtractor zapamietuje kolejnosc kolumn
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix, hstack

sentiment = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """
    Zwraca sentyment tekstu miary (compound, pos, neg, neu)
    """
    if not text or pd.isna(text):
        return {
            'sentiment_compound': 0.0,
            'sentiment_pos': 0.0,
            'sentiment_neg': 0.0,
            'sentiment_neu': 0.0,
        }

    score = sentiment.polarity_scores(text)
    return {
        'sentiment_compound': score['compound'],
        'sentiment_pos': score['pos'],
        'sentiment_neg': score['neg'],
        'sentiment_neu': score['neu'],
    }

# Potrzebne aliasy dla kompatybilnosci z predictor.py
extract_all_features = get_sentiment

def extract_features_batch(texts):
    """ Ekstrakcja sentymentu z listy tekstow. Dla treningu modelu"""
    print(f"Ekstrakcja cech z {len(texts)} tekstow..")
    features = [get_sentiment(text) for text in texts]
    print("Zakonczone")
    return pd.DataFrame(features)

def combine_features(tfidf_matrix, features_df):
    """ Laczenie macierzy TF-IDF z sentymentem"""
    if isinstance(features_df, pd.DataFrame):
        features_sparse = csr_matrix(features_df.values)
    else:
        features_sparse = csr_matrix(features_df)
    return hstack([tfidf_matrix, features_sparse]).tocsr()

class FeatureExtractor:
    """ Klasa dla ekstrakcji cech - zapamietuje kolejnosc kolumn"""

    def __init__(self):
        self.feature_columns = None

    def fit_transform(self, texts):
        """ Ekstrakcja cechy i zapamietanie kolejnosci kolumn dla treningu"""
        df = extract_features_batch(texts)
        self.feature_columns = df.columns.tolist()
        return df

    def transform(self, texts):
        """ Ekstrakcja cechy i zapamietanie kolejnosci kolumn dla predykcji"""
        df = extract_features_batch(texts)
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
                df = df[self.feature_columns]
        return df

