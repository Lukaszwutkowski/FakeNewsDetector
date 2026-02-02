"""
Skrypt ma za zadanie eksploracje danych
w celu zrozumienia struktury oraz charakterystyki zbioru danych
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Rozwiazuje problem sciezki plikow
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "news_articles.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed")

def load_data(filepath):
    "Wczytuje dane z pliku csv"
    if not os.path.isfile(filepath):
        print(f"Plik {filepath} nie istnieje.")
        return None

    print(f"Wczytywanie danych z pliku {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dane wczytane: W pliku znaleziono {len(df)} rekordow.")
    return df

def basic_info(df):
    "Podstawowe informacje o zbiorze danych"
    print("=" * 50)
    print("Podstawowe informacje o zbiorze danych:")
    print("=" * 50)
    print(f"Liczba rekordow: {len(df)}")
    print(f"Nazwa kolumn: {list(df.columns.values)}")
    print(f"Typ danych:\n {df.dtypes}")
    print(f"Pierwsze 5 rekordow: \n{df.head(5)}")
    print("=" * 50)

def missing_values(df):
    "Wykresy z brakujacych wartosci"
    print("=" * 50)
    print("Wykresy z brakujacych wartosci:")
    print("=" * 50)

    missing = df.isnull().sum()
    missing_percent = (missing/len(df))*100

    missing_df = pd.DataFrame({
        'Brakujace': missing, 'Procent': missing_percent})

    print(missing_df[missing_df['Brakujace'] > 0])

    if missing.sum() == 0:
        print("Brak brakujacych wartosci.")


def main():
    # Wczytuje dane
    df = load_data(DATA_PATH)

    basic_info(df)
    missing_values(df)


# print(os.getcwd()) -> potrzebny do sprawdzenia relative path

if __name__ == "__main__":
    main()

