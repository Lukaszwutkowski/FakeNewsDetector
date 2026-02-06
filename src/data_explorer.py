"""
Wersja 1.0

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

Wersja 1.1
Refaktoryzacja objela - Struktura OOP:
- DataExplorer jako klasa bazowa
- OldDatasetExplorer - analiza pliku csv z poprzedniej wersji
- NewDatasetExplorer - analiza pliku csv z nowej wersji

Dodatkowo refaktoryzacja objela uporzadkowanie sciezek plikow i katalogow
za pomoca Path z biblioteki pathlib oraz pliku konfiguracyjnego yaml
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from utils.config import Paths, Files


# Rozwiazuje problem sciezki plikow
# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) --> POPRZEDNIE ROZWIAZANIE


class DataExplorer:
    """
    Klasa bazowa dla analizy danych.
    Ma na celu zrozumienie struktury oraz charakterystyki zbioru danych.
    """
    def __init__(self, name: str = "Dataset"):
        self.df = None
        self.name = name
        self.output_path = Paths.DATA_PROCESSED / self.name # --> rozwiazanie problemu z nadpisywaniem wykresow
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_data(self, filepath):
        "Wczytuje dane z pliku csv. Korzystam z pandas DataFrame dla wygodnego dostepu do danych."
        filepath = Path(filepath)
        if not filepath.is_file():
            print(f"Plik {filepath} nie istnieje.")
            return None

        print(f"Wczytywanie danych z pliku {filepath}")
        self.df = pd.read_csv(filepath)
        print(f"Dane wczytane: W pliku znaleziono {len(self.df)} rekordow.")
        return self.df

    def basic_info(self):
        """Podstawowe informacje o zbiorze danych. Ma na celu uzyskac informacje:
        len(df), df.columns, df.dtypes, df.head()"""
        print("=" * 50)
        print("Podstawowe informacje o zbiorze danych:")
        print("=" * 50)
        print(f"Liczba rekordow: {len(self.df)}")
        print(f"Nazwa kolumn: {list(self.df.columns.values)}")
        print(f"Typ danych:\n {self.df.dtypes}")
        print(f"Pierwsze 5 rekordow: \n{self.df.head(5)}")
        print("=" * 50)

    def missing_values(self):
        """Brakujace wartosci. Ma na celu sprawdzic braki danych"""
        print("=" * 50)
        print("Wykresy z brakujacych wartosci:")
        print("=" * 50)

        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100

        missing_df = pd.DataFrame({
            'Brakujace': missing, 'Procent': missing_percent})

        print(missing_df[missing_df['Brakujace'] > 0])

        if missing.sum() == 0:
            print("Brak brakujacych wartosci.")

    def target_analysis(self):
        """
        Analiza zmiennej celu (label). Ma na celu uzyskanie rozkladu klas w zbiorze danych.
        Poznanie jakie sa klasy, jak sa zbalansowane - proporcje
        """
        print("=" * 50)
        print("Analiza zmiennej celu (label):")
        print("=" * 50)

        label_count = self.df['label'].value_counts()
        label_percent = (label_count / len(self.df) * 100).round(2)

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
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'label_distribution.png'))
        print("Zapisano wykres do pliku label_distribution.png")
        plt.close()

    def text_length_analysis(self):
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
        self.df['text_length'] = self.df['text'].fillna('').str.len()
        self.df['title_length'] = self.df['title'].fillna('').str.len()

        print("\n Dlugosc artykulu w zbiorze danych:")
        print(self.df[['text_length', 'title_length']].describe().round(2))

        print("\n Dlugosc tytulu:")
        print(self.df['title_length'].describe().round(2))

        # Funkcja do porownania dlugosci dla Real oraz Fake news
        print("\n Porownanie dlugosci artykulu dla Real oraz Fake news:")
        print(self.df.groupby('label')['text_length'].mean().round(2))

        # Rysowanie wykresu z wynikami
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram dlugosci tekstu
        for label, color in [('Real', 'blue'), ('Fake', 'red')]:
            subset = self.df[self.df['label'] == label]
            if len(subset) > 0:
                subset['text_length'].hist(ax=axes[0], bins=50, alpha=0.7, label=label, color=color)
        axes[0].legend()
        axes[0].set_title(f'Histogram dlugosci tekstu - {self.name}')
        axes[0].set_xlabel('Liczba znakow')
        axes[0].set_ylabel('Liczba artykow')
        axes[0].set_xlim(0, 2000)

        # Analogicznie histogram dla tytulu
        for label, color in [('Real', 'blue'), ('Fake', 'red')]:
            subset = self.df[self.df['label'] == label]
            if len(subset) > 0:
                subset['title_length'].hist(ax=axes[1], bins=50, alpha=0.7, label=label, color=color)
        axes[1].legend()
        axes[1].set_title('Histogram dlugosci tytulu')
        axes[1].set_xlabel('Liczba znakow')
        axes[1].set_ylabel('Liczba artykow')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'text_length_distribution.png'))
        print("Zapisano wykres do pliku text_length_distribution.png")
        plt.close()

    def _safe_name(self):
        """Zwraca bezpieczny string dla nazwy pliku."""
        return "".join(c if c.isalnum() else "_" for c in self.name).lower()

    def run_all(self):
        """Uruchamia wszystkie funkcje analizy danych."""
        self.basic_info()
        self.missing_values()
        self.target_analysis()
        self.text_length_analysis()

class OldDatasetExplorer(DataExplorer):
    """
    Analiza pliku csv z poprzedniej wersji.
    """
    def __init__(self):
        super().__init__("old")
        self.filepath = Files.NEWS_ARTICLES

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Wczytuje dane z pliku csv. Korzystam z pandas DataFrame dla wygodnego dostepu do danych.
        """
        return super().load_data(filepath or self.filepath)

    def languages_analysis(self):
        """
        Analiza jazykow. Ma na celu uzyskanie rozkladu jazykow w zbiorze danych.
        Celem ogolnym jest budowanie modelu na bazie jezyka angielskiego
        -popularnosc, wieksza proporcja fake news. Jest to tez specyficzne dla
        tego datasetu
        """
        print("=" * 50)
        print("Analiza jazykow:")
        print("=" * 50)

        lang_count = self.df['language'].value_counts()
        print("\nRozklad jazykow w zbiorze danych:")
        print(lang_count)

    def run_all(self):
        """
        Uruchamia wszystkie funkcje analizy danych wraz z analiza jezyka
        """
        super().run_all()
        self.languages_analysis()

class NewDatasetExplorer(DataExplorer):
    """
    Analiza pliku csv z nowej wersji.
    """

    def __init__(self):
        super().__init__("new")
        self.fake_path = Files.DATA_FAKE
        self.true_path = Files.DATA_TRUE

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Celem jest wczytanie plikow z danymi Fake oraz True.
        Pliki zostana polaczone
        """
        print(f"Wczytywanie danych z plikow Fake oraz True z folderu newDataset")

        if not self.fake_path.exists() or not self.true_path.exists():
            print("Nie znaleziono plikow Fake.csv lub True.csv w folderze newDataset")
            return None

        # Wczytywanie plikow
        df_fake = pd.read_csv(self.fake_path)
        df_fake['label'] = 'Fake'
        print(f"Wczytano {len(df_fake)} rekordow z pliku Fake.csv")

        df_true = pd.read_csv(self.true_path)
        df_true['label'] = 'Real'
        print(f"Wczytano {len(df_true)} rekordow z pliku True.csv")

        # Laczenie dwoch plikow
        self.df = pd.concat([df_fake, df_true], ignore_index=True)
        print(f"Polaczono dane z plikow Fake.csv oraz True.csv w zbiorze danych.\n "
              f"Otrzymano {len(self.df)} rekordow.")

        return self.df

    def subject_analysis(self):
        """
        Analiza tematow. Ma na celu uzyskanie rozkladu tematow w zbiorze danych.
        Jest to specyficzne dla tego datasetu
        """

        if 'subject' not in self.df.columns:
            print("Nie znaleziono kolumny subject w zbiorze danych.")
            return

        print("=" * 50)
        print(f"Analiza tematow: {self.name}")
        print("=" * 50)

        subject_count = self.df['subject'].value_counts()
        print("\nRozklad tematow w zbiorze danych:")
        print(subject_count)

        # Rozklad tematow
        print("\nRozklad tematow w zbiorze danych:")
        print(self.df.groupby(['label', 'subject']).size().unstack(fill_value=0))

    def run_all(self):
        """
        Uruchamia wszystkie funkcje analizy danych wraz z analiza tematu
        """
        super().run_all()
        self.subject_analysis()

def main():
    """
    Glowna funkcja, Zmieniona na potrzeby refaktoryzacji.
    Funkcja uruchamia analizy dla zbioru danych obu wersji.
    """
    print("START main()")
    print("Paths.DATA_PROCESSED =", Paths.DATA_PROCESSED)
    print("Files.NEWS_ARTICLES =", Files.NEWS_ARTICLES)

    print("=" * 50)
    print("Witaj w skrypcie eksploracji danych!")
    print("=" * 50)

    # Analiza poprzedniego datasetu
    print("=" * 50)
    print("Analiza poprzedniego datasetu:")
    print("=" * 50)

    old_explorer = OldDatasetExplorer()
    if old_explorer.load_data() is not None:
        old_explorer.run_all()

    # Analiza nowego datasetu
    print("=" * 50)
    print("Analiza nowego datasetu:")
    print("=" * 50)

    new_explorer = NewDatasetExplorer()
    if new_explorer.load_data() is not None:
        new_explorer.run_all()

    print("=" * 50)
    print("Analiza Zakonczona. Wykresy zapisano do folderu: {Paths.DATA_PROCESSED}".format(Paths=Paths))
    print("=" * 50)


# print(os.getcwd()) -> potrzebny do sprawdzenia relative path

if __name__ == "__main__":
    main()

