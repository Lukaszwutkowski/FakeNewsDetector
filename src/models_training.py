"""
Wersja 1.0 - model na poprzednim datasecie.
Skrypt odpowiedzialny za trenowanie modelu

Wersja 1.1 - model na nowym i starym datasecie

Wersja 1.2 - refaktoryzacja skryptu. Modul zawiera teraz wspolne funkcje
i klasy do przetwarzania danych i trenowania modeli.
- Ladowanie danych z pliku csv
- Przygotowanie danych do trenowania
- Dzielenie danych na zbiory treningowe i testowe
- Wektoryzacja tekstu
- Trenowanie modeli
- Zapis modeli w pliku
- Ocena modeli na zbiorze testowym
"""


import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_explorer import NewDatasetExplorer
from src.models import get_model, get_available_models
from utils.config import Paths, ModelConfig, dir_exists
from utils.feature_engineering import FeatureExtractor, combine_features
from utils.text_processing import text_preprocessing


def preprocessing_data(df):
    "Przygotowanie danych do trenowania"
    print("Przygotowywanie danych do trenowania...")

    # Polaczenie tytulu oraz tekstu
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Preprocessing
    df['processed_text'] = df['combined_text'].apply(text_preprocessing)

    # Zamiana na wartosci numeryczne odpowiadajace Real=0 i Fake=1
    df['label_numeric'] = df['label'].map({'Real': 0, 'Fake': 1})

    # Wyszukanie rekordow z brakujacymi etykietami i  usuniecie ich
    before = len(df)
    df = df.dropna(subset=['label_numeric'])
    after = len(df)

    if before != after:
        print(f"Usunieto {before - after} rekordow z brakujacymi etykietami.")

    print(f"Dane po przetworzeniu: W pliku znaleziono {len(df)} rekordow.")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Dzielenie danych na zbiory treningowe i testowe:"""
    print("Dzielenie danych na zbiory treningowe i testowe...")

    x = df['processed_text']
    y = df['label_numeric']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"Rozmiar zbioru treningowego: {len(x_train)} rekordow.")
    print(f"Rozmiar zbioru testowego: {len(x_test)} rekordow.")

    return x_train, x_test, y_train, y_test

def vectorize_text(x_train, x_test, max_features=5000):
    """
    Wektoryzacja tekstu: Zastosowano TF-IDF do wektoryzacji tekstu.
    Celem jest przeksztalcenie tekstu na wektor liczb
    """
    print("Wektoryzacja tekstu...")

    vectorizer = TfidfVectorizer(
        max_features=max_features, stop_words='english', ngram_range=(1, 2))

    # Zbior treningowy - dopasowanie oraz transformacja
    x_train_tfidf = vectorizer.fit_transform(x_train)

    # Zbior testowy = transformacja
    x_test_tfidf = vectorizer.transform(x_test)

    print(f"Rozmiar zbioru wektoryzacji treningowego: {x_train_tfidf.shape}")
    print(f"Rozmiar zbioru wektoryzacji testowego: {x_test_tfidf.shape}")

    return vectorizer, x_train_tfidf, x_test_tfidf

def train_model(model, x_train, y_train, model_name="Model"):
    """
    Trenowanie pojedynczego modelu. Dodanie metody model_name, aby zmienic nazwe modelu.
    Dodanie metody start_time, aby zobaczyc czas trwania trenowania modelu.
    """
    print(f"Trenowanie modelu: {model_name}")

    start_time = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start_time
    print(f"Czas trenowania modelu: {train_time}")

    return model, train_time

def save_model(model, vectorizer, feature_extractor, model_dir, model_name="model"):
    """Zapis modelu w pliku."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Zapis modelu w pliku...")

    model_path = model_dir / "model.joblib"
    vectorizer_path = model_dir / "vectorizer.joblib"
    feature_extractor_path = model_dir / "feature_extractor.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(feature_extractor, feature_extractor_path)

    print(f"Model zapisany w pliku: {model_path}")
    print(f"Vectorizer zapisany w pliku: {vectorizer_path}")

    # Dodatkowe informacje o modelu zapisane do pliku txt dla oceny
    info_path = model_dir / "model_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Typ: {type(model).__name__}\n")

def save_confusion_metric(cm, model_name, out_dir):
    """
    Funkcja zapisuje macierze pomylek. Wyodrebniona jako osobna funkcja z
    funkcji evaluate_model.
    """
    # Wykres macierzy pomylek
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Macierz pomylek modelu {model_name}', fontsize=14, pad=20)
    plt.colorbar()
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Wartosci na wykresie
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)

    plt.ylabel('Rzeczywista klasa')
    plt.xlabel("Przewidziana klasa")
    plt.tight_layout()

    filepath = out_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(filepath)
    plt.close()

def evaluate_model(model, x_test, y_test, model_name="Model", out_dir=None):
    """
    Ocena modelu na zbiorze testowym.
    - Restrukturyzacja objela - dodanie parametru out_dir na None,
    aby funkcja mogla dzialac bez koniecznosci podawania parametru out_dir.
    - Dodanie model_name, aby wybrac model.
    """
    print(f"Ocena modelu na zbiorze testowym: {model_name}")

    # Predykcja
    y_pred = model.predict(x_test)

    # Metryki
    metrics = {
       'model': model_name,
       'accuracy': accuracy_score(y_test, y_pred),
       'precision': precision_score(y_test, y_pred),
       'recall': recall_score(y_test, y_pred),
       'f1': f1_score(y_test, y_pred)
    }

    print(
        f"\n\n======================\n"
        f"Metryki oceny modelu {model_name}:\n"
        f"Accuracy: {metrics['accuracy']:.4f}({metrics['accuracy']*100:.2f}%)\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1: {metrics['f1']:.4f}\n"
        f"======================\n"
    )

    # Macierz pomylek
    cm = confusion_matrix(y_test, y_pred)
    print(
        "Macierz pomylek:\n",
        pd.DataFrame(pd.DataFrame(cm, index=["Real", "Fake"], columns=["Real", "Fake"]))
    )

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_metric(cm, model_name, out_dir)

    return metrics

def load_trained_model(model_dir=None):
    """
    Odpowiada za ladowanie modelu z pliku. Wybiera najlepszy model z katalogu best_model.
    """
    if model_dir is None:
        model_dir = Paths.MODELS / ModelConfig.ACTIVE_MODEL

    """
    DEBUGOWANIE SCIEZKI DO KATALOGU MODELU
    print(f"\n{'=' * 60}")
    print(f"DEBUG - Sciezka do katalogu modelu:")
    print(f"  model_dir: {model_dir}")
    print(f"  model_dir (absolute): {model_dir.resolve()}")
    print(f"{'=' * 60}\n")"""

    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    vectorizer_path = model_dir / "vectorizer.joblib"
    feature_extractor_path = model_dir / "feature_extractor.joblib"

    if not model_path.exists() or not vectorizer_path.exists():
        print(f"Model {model_dir} nie zostal znaleziony.")
        return None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    feature_extractor = joblib.load(feature_extractor_path)

    return model, vectorizer, feature_extractor


def _plot_metrics(results_df, out_dir, save=True):
    """
    Wykres slupkowy porownujacy metryki accuracy, precision, recall i f1.
    """
    metrics = [m for m in ['accuracy', 'precision', 'recall', 'f1'] if m in results_df.columns]
    if not metrics:
        print("Brak danych do wykresu.")
        return

    model_names = results_df['model'].tolist()
    plot_df = results_df.set_index('model')[metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot.bar(ax=ax, rot=0, width=0.8)
    ax.set_title("Porownanie metryk accuracy, precision, recall i f1")
    ax.set_xlabel("Model")
    ax.set_ylabel("Wartosc")
    ax.set_ylim(0, 1)
    ax.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        filepath = out_dir / "metrics_comparison.png"
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


def _plot_f1(results_df, out_dir, save=True):
    """
    Ranking F1 dla modeli.
    """
    if 'f1' not in results_df.columns:
        print("Brak danych do wykresu.")
        return

    sorted_df = results_df.sort_values(by='f1', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df['model'], sorted_df['f1'])
    plt.title("Ranking F1 dla modeli")
    plt.xlabel("F1")
    plt.ylabel("Model")
    plt.xlim(0, 1)

    for i, v in enumerate(sorted_df['f1']):
        plt.text(v + 0.02, i, f'{v:.4f}', ha='left', va='center')

    plt.tight_layout()

    if save:
        filepath = out_dir / "f1_ranking.png"
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()

def create_comparision_plots_for_models(results_df, out_dir, save=True):
    """
    Tworzenie wykresow porownawczych dla modeli. Ma to na celu ewaluacje modeli.
    Wyniki do przedstawienia na obronie!!
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metryki
    _plot_metrics(results_df, out_dir, save=save)

    # Ranking F1
    _plot_f1(results_df, out_dir, save=save)

def compare_models_and_choose_best():
    """
    Funkcja uruchamia, trenuje oraz porownuje dostepne modele.
    Na podstawie metryk wybiera najlepszy model.
    """
    dir_exists() # --> upewnienie sie ze sa katalogi

    print("=" * 50)
    print("Porownanie modeli")
    print("=" * 50)

    # Ladowanie danych - korzystam z nowego Datasetu (NewDataSetExplorer)
    # wybor po analizie danych, wiecej rekordow.
    explorer = NewDatasetExplorer()
    df = explorer.load_data()
    if df is None:
        print("Nie udalo sie wczytac danych.")
        return

    df = preprocessing_data(df)
    x_train, x_test, y_train, y_test = split_data(df)

    """ =========== TF-IDF ========="""
    vectorizer, x_train_tfidf, x_test_tfidf = vectorize_text(x_train, x_test)

    """ ======= VADER ========"""
    # Analiza sentymentu
    feature_extractor = FeatureExtractor()
    train_sentiment = feature_extractor.fit_transform(x_train)
    test_sentiment = feature_extractor.transform(x_test)

    """ ======= TF-IDF + VADER ======="""
    x_train_final = combine_features(x_train_tfidf, train_sentiment)
    x_test_final = combine_features(x_test_tfidf, test_sentiment)

    """ ====== Debug dla upewnienia sie ze +4 cechy w final ===== """
    print("Shape train TF-IDF", x_train_tfidf.shape)
    print("Shape train FINAL", x_train.final.shape)

    results = []

    # Katalog wymagany do zapisania wynikow, potrzebny do metody tworzenia wykresu
    comparison_dir = Paths.MODELS / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    """ ========= Trenowanie wszystkich modeli ========="""
    for model_name in get_available_models():
        print(f"\nTrenowanie modelu: {model_name}")
        model = get_model(model_name)
        model, train_time = train_model(model, x_train_final, y_train, model_name)
        metrics = evaluate_model(model, x_test_final, y_test, model_name)
        metrics['czas_trenowania'] = f"{train_time:.2f}s"
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # zapis wynikow do pliku csv
    results_csv_path = comparison_dir / "model_comparison_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Wyniki porownawcze zapisane do pliku: {results_csv_path}")

    # Tworzenie wykresow porownawczych
    create_comparision_plots_for_models(results_df, comparison_dir, save=True)
    print("Wszystkie modele porownane i wyniki zapisane w plikach.")

    """ ======== Wybor najlepszego modelu na podstawie porownania ===== """
    best_idx = results_df['f1'].idxmax()
    best_model = results_df.loc[best_idx]
    best_model_name = best_model['model']
    print(f"\nNajlepszy model: {best_model_name}")

    """ ==== Ponowne trenowanie najlepszego modelu na pelnych danych ==== """
    best_model = get_model(best_model_name)
    best_model, final_train_time = train_model(best_model, x_train_final, y_train, best_model_name)

    # Zapisanie modelu w katalogu best_model
    best_model_dir = Paths.MODELS / "best_model"
    save_model(best_model, vectorizer, feature_extractor, best_model_dir, best_model_name)
    print(f"Najlepszy model zapisany w katalogu: {best_model_dir}")

    return results_df, best_model_name

def main():
    print("Wersja 1.2 - refaktoryzacja skryptu.")
    print("Trenowanie modeli")
    print("=" * 50)

    compare_models_and_choose_best()

if __name__ == "__main__":
    main()



