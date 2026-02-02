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

def target_analysis(df):
    "Analiza zmiennej celu (label)"
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

    # Przeniesienie warrtosci na slupki
    for bar, count in zip(bars, label_count.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'label_distribution.png'))
    print("Zapisano wykres do pliku label_distribution.png")
    plt.close()


def main():
    """ Glowna funkcja, podzial na podfunkcje: Wstep, Wczytanie danych, Analiza """
    print("=" * 50)
    print("Witaj w skrypcie eksploracji danych!")
    print("=" * 50)

    # Tworzenie folderu output jezeli nie istnieje
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Wczytuje dane
    df = load_data(DATA_PATH)

    # Analiza
    basic_info(df)
    missing_values(df)
    target_analysis(df)


# print(os.getcwd()) -> potrzebny do sprawdzenia relative path

if __name__ == "__main__":
    main()

