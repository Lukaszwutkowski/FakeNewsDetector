"""
Skrypt ma za zadanie eksploracje danych
w celu zrozumienia struktury oraz charakterystyki zbioru danych
Chce poznac:
ilosc rekordow,
nazwy kolumn,
typ danych,
pierwsze 5 rekordow,
brakujace wartosci,
rozklad klas,
wykresy
Podsumowanie:
- podstawowe informacje o zbiorze danych
- wykresy z brakujacych wartosci
- analiza zmiennej celu (label)
- analiza dlugosci tekstu
- analiza jezykow
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Rozwiazuje problem sciezki plikow
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "news_articles.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "processed")

def load_data(filepath):
    "Wczytuje dane z pliku csv. Korzystam z pandas DataFrame dla wygodnego dostepu do danych."
    if not os.path.isfile(filepath):
        print(f"Plik {filepath} nie istnieje.")
        return None

    print(f"Wczytywanie danych z pliku {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dane wczytane: W pliku znaleziono {len(df)} rekordow.")
    return df

def basic_info(df):
    """Podstawowe informacje o zbiorze danych. Ma na celu uzyskac informacje:
    len(df), df.columns, df.dtypes, df.head()"""
    print("=" * 50)
    print("Podstawowe informacje o zbiorze danych:")
    print("=" * 50)
    print(f"Liczba rekordow: {len(df)}")
    print(f"Nazwa kolumn: {list(df.columns.values)}")
    print(f"Typ danych:\n {df.dtypes}")
    print(f"Pierwsze 5 rekordow: \n{df.head(5)}")
    print("=" * 50)

def missing_values(df):
    """Brakujace wartosci. Ma na celu sprawdzic braki danych"""
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

def target_analysis(df):
    """
    Analiza zmiennej celu (label). Ma na celu uzyskanie rozkladu klas w zbiorze danych.
    Poznanie jakie sa klasy, jak sa zbalansowane - proporcje
    """
    print("=" * 50)
    print("Analiza zmiennej celu (label):")
    print("=" * 50)

    label_count = df['label'].value_counts()
    label_percent = (label_count/len(df)*100).round(2)

    print("\nRozklad klas w zbiorze danych:")
    for label, count in label_count.items():
        percent = label_percent[label]
        print(f"{label}: {count} ({percent}%)")

    # Budowanie wykresu
    plt.figure(figsize=(10, 6))
    colors = ['blue' if l == 'Real' else 'red' for l in label_count.index]
    bars = plt.bar(label_count.index, label_count.values, color=colors)

    plt.title('Rozklad klas w zbiorze danych')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba rekordow')

    # Przeniesienie wartosci na slupki
    for bar, count in zip(bars, label_count.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'label_distribution.png'))
    print("Zapisano wykres do pliku label_distribution.png")
    plt.close()

def text_length_analysis(df):
    """
    Analiza dlugosci tekstu. Ma na celu uzyskanie rozkladu dlugosci tekstu w zbiorze danych.
    Poznanie jakie sa dlugosci artykow, dlugosci tytulow.
    Porownanie dlugosci artykow dla Real oraz Fake news.
    Dlugie teksty - moze spowolnic model
    Krotsze teksty - malo informacji do klasyfikacji
    """
    print("=" * 50)
    print("Analiza dlugosci tekstu:")
    print("=" * 50)

    # Funkcja oblicza dlugosc tekstu poprzez zliczanie liczby znakow dla tekstu oraz tytulu
    df['text_length'] = df['text'].fillna('').str.len()
    df['title_length'] = df['title'].fillna('').str.len()

    print("\n Dlugosc artykulu w zbiorze danych:")
    print(df[['text_length', 'title_length']].describe().round(2))

    print("\n Dlugosc tytulu:")
    print(df['title_length'].describe().round(2))

    # Funkcja do porownania dlugosci dla Real oraz Fake news
    print("\n Porownanie dlugosci artykulu dla Real oraz Fake news:")
    print(df.groupby('label')['text_length'].mean().round(2))

    # Rysowanie wykresu z wynikami
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram dlugosci tekstu
    df[df['label'] == 'Real']['text_length'].hist(
        ax=axes[0], bins=50, alpha=0.7, label='Real', color='blue')
    df[df['label'] == 'Fake']['text_length'].hist(
        ax=axes[0], bins=50, alpha=0.7, label='Fake', color='red'
    )
    axes[0].legend()
    axes[0].set_title('Histogram dlugosci tekstu')
    axes[0].set_xlabel('Liczba znakow')
    axes[0].set_ylabel('Liczba artykow')
    axes[0].set_xlim(0, 2000)

    # Analogicznie histogram dla tytulu
    df[df['label'] == 'Real']['title_length'].hist(
        ax=axes[1], bins=50, alpha=0.7, label='Real', color='blue')
    df[df['label'] == 'Fake']['title_length'].hist(
        ax=axes[1], bins=50, alpha=0.7, label='Fake', color='red'
    )
    axes[1].legend()
    axes[1].set_title('Histogram dlugosci tytulu')
    axes[1].set_xlabel('Liczba znakow')
    axes[1].set_ylabel('Liczba artykow')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'text_length_distribution.png'))
    print("Zapisano wykres do pliku text_length_distribution.png")
    plt.close()

    return df

def languages_analysis(df):
    """
    Analiza jazykow. Ma na celu uzyskanie rozkladu jazykow w zbiorze danych.
    Celem ogolnym jest budowanie modelu na bazie jezyka angielskiego
    -popularnosc, wieksza proporcja fake news.
    """
    print("=" * 50)
    print("Analiza jazykow:")
    print("=" * 50)

    lang_count = df['language'].value_counts()
    print("\nRozklad jazykow w zbiorze danych:")
    print(lang_count)

def main():
    """ Glowna funkcja, podzial na podfunkcje: Wstep, Wczytanie danych, Analiza """
    print("=" * 50)
    print("Witaj w skrypcie eksploracji danych!")
    print("=" * 50)

    # Tworze folder output jezeli nie istnieje
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Wczytuje dane
    df = load_data(DATA_PATH)

    # Analiza
    basic_info(df)
    missing_values(df)
    target_analysis(df)
    df = text_length_analysis(df)
    languages_analysis(df)


# print(os.getcwd()) -> potrzebny do sprawdzenia relative path

if __name__ == "__main__":
    main()

