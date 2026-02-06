

import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils.text_processing import text_preprocessing
from utils.config import Files, Paths

def load_data(filepath):
    "Wczytuje dane z pliku csv"
    filepath = Path(filepath)
    if not filepath.is_file():
        print(f"Plik {filepath} nie istnieje.")
        return None

    print(f"Wczytywanie danych z pliku {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dane wczytane: W pliku znaleziono {len(df)} rekordow.")
    return df

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
    df = df.dropna(subset=['label_numeric'])

    print(f"Dane po przetworzeniu: W pliku znaleziono {len(df)} rekordow.")
    return df

def split_data(df, test_size=0.2):
    """Dzielenie danych na zbiory treningowe i testowe:"""
    print("Dzielenie danych na zbiory treningowe i testowe...")

    x = df['processed_text']
    y = df['label_numeric']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, stratify=y)

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

def train_model(x_train_tfidf, y_train):
    """Trenowanie modelu. Logistyczna regresja."""
    print("Trenowanie modelu...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        class_weight='balanced'
    )

    model.fit(x_train_tfidf, y_train)
    return model

def save_model(model, vectorizer, model_dir: Path):
    """Zapis modelu w pliku."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Zapis modelu w pliku...")

    model_path = model_dir / "model.joblib"
    vectorizer_path = model_dir / "vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Model zapisany w pliku: {model_path}")
    print(f"Vectorizer zapisany w pliku: {vectorizer_path}")

def evaluate_model(model, x_test_tfidf, y_test, run_name: str):
    """
    Ocena modelu na zbiorze testowym.
    - Dodanie metody run_name ktora pozwala na wybor datasetu
    """
    out_dir = Paths.DATA_PROCESSED / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Ocena modelu na zbiorze testowym...")

    # Predykcja
    y_pred = model.predict(x_test_tfidf)

    # Metryki
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(
        f"Metryki oceny modelu na zbiorze testowym:\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1: {f1:.4f}"
    )

    # Macierz pomylek
    cm = confusion_matrix(y_test, y_pred)
    print(
        "Macierz pomylek:\n",
        pd.DataFrame(cm, index=["Real", "Fake"], columns=["Real", "Fake"])
    )

    # Wykres macierzy pomylek
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Macierz pomylek')
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
    plt.savefig(out_dir / "confusion_matrix.png")
    print(f"Dokonano oceny modelu na zbiorze testowym. "
          f"Dokonano {accuracy*100:.2f}% prawidlowych predykcji.")
    plt.close()

    return accuracy, precision, recall, f1

def train_pipeline(df: pd.DataFrame, run_name: str):
    "Wszystkie kroki przetwarzania danych do trenowania modelu"
    print("=" * 50)
    print(f"Wykonywanie pipeline dla datasetu: {run_name}")
    print("=" * 50)

    df = preprocessing_data(df)
    x_train, x_test, y_train, y_test = split_data(df)
    vectorizer, x_train_tfidf, x_test_tfidf = vectorize_text(x_train, x_test)
    model = train_model(x_train_tfidf, y_train)
    save_model(model, vectorizer, Paths.MODELS / run_name)
    evaluate_model(model, x_test_tfidf, y_test, run_name)

def main():
    # Model na poprzednim datasecie
    df_old = load_data(Files.NEWS_ARTICLES)
    if df_old is not None:
        train_pipeline(df_old, run_name="old_dataset")

    # Model na nowym datasecie

    # Model na nowym datasecie
    df_fake = pd.read_csv(Files.DATA_FAKE)
    df_fake['label'] = 'Fake'
    df_true = pd.read_csv(Files.DATA_TRUE)
    df_true['label'] = 'Real'
    df_new = pd.concat([df_fake, df_true])
    train_pipeline(df_new, run_name="new_dataset")


if __name__ == "__main__":
    main()