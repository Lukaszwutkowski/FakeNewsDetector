"""
Modul ma za zadanie wczytanie danych z pliku csv. Przeniesienie logiki wczytywania
tutaj unika duplikacji kodu miedzy models_training oraz data_explorer.
"""
import pandas as pd
from pathlib import Path

def load_data_from_csv(filepath):
    "Wczytuje dane z pliku csv"
    filepath = Path(filepath)
    if not filepath.is_file():
        print(f"Plik {filepath} nie istnieje.")
        return None

    print(f"Wczytywanie danych z pliku {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dane wczytane: W pliku znaleziono {len(df)} rekordow.")
    return df