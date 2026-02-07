import re

import pandas as pd


def text_preprocessing(text):
    """
    Przygotowanie tekstu do modelowania:
    W celu uzyskania najlepszych wynikow przygotuje tekst do modelowania.
    -- Usuniecie znakow specjalnych -- zamiana na male litery -- usuniecie nadmiarowych spacji
    Usuwanie 'szumu'.
    """

    if pd.isna(text):
        return ""

    # Wykorzystanie nowej funkcji usuwania metadanych zrodlowych
    text = normalize_source(text)

    text = re.sub(r'[^a-zA-Z0-9\s]', "", text)
    text = text.lower()
    text = ' '.join(text.split())

    return text

def normalize_source(text):
    """
    Usuniecie zrodla, pochodzenia artykulu. Model nie wymaga tego parametru.
    Co wiecej moze to powodowac bledy w predykcji.
    """
    if pd.isna(text):
        return ""

    # Cel to usuniecie naglowkow z artykulu typu miasto-zrodlo ktore pojawiaja sie na poczatku
    # Chce zapobiec przez model kojarzenia okreslonych slow z konkretnym zrodlem
    # Liczy sie sens zdania dlatego to nalezy usunac
    text = re.sub(r'^[A-Z]{2,}[A-Z\s,]*\([^)]+\)\s*[-–—]\s*',"", text)

    # Usuwanie zrodla ktore pojawia sie w nawiasie na poczatku
    text = re.sub(r'^\([^)]+\)\s*[-–—]?\s*',"", text)

    # Usuwanie nazw agencji w calym tekscie
    agencies = ['AP','CNN', 'BBC', 'Fox News', 'Reuters', 'The Guardian', 'The Washington Post', 'Bloomberg',
                'New York Times', 'The New York Times', 'The Wall Street Journal', 'The Atlantic']
    for agency in agencies:
        text = re.sub(rf'\b{agency}\b', "", text, flags=re.IGNORECASE)

    return text