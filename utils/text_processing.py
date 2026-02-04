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

    text = re.sub(r'[^a-zA-Z0-9\s]', "", text)
    text = text.lower()
    text = ' '.join(text.split())

    return text