"""
Dodatkowy modul nie majacy ogolnego znaczenia w dzialaniu projektu.
Stworzony na potrzeby edukacyjne, do pracy inzynierskiej w celu
udowodnienia twierdzen hipotez.
"""
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data_explorer import NewDatasetExplorer
from src.models import get_model
from utils.feature_engineering import extract_features_batch
from utils.text_processing import text_preprocessing


def load_data():
    """
    Ladowanie danych przez NewDatasetExplorer
    """
    explorer = NewDatasetExplorer()
    df = explorer.load_data()
    df['label'] = df['label'].map({'Real': 0, 'Fake': 1})
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    return df

def text_preprocessing_without_normalize(text):
    return text_preprocessing(text, use_normalization=False)

def evaluate(x_train, x_test, y_train, y_test, model_name="RandomForestClassifier"):
    """ Trening Random Forest zwraca F1-score"""
    model = get_model(model_name)
    model.fit(x_train, y_train)
    return f1_score(y_test, model.predict(x_test))

def vectorize(x_train, x_test):
    """ Wektoryzacja TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    return vectorizer.fit_transform(x_train).toarray(), vectorizer.transform(x_test).toarray()

def main():
    print("=" * 50)
    print("Hipotezy:")
    print("=" * 50)

    df = load_data()
    x_raw, x_test_raw, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    print("=" * 50)
    print(f"Dane: {len(df)} artykulow")


    """============== Hipoteza (H1) ================"""
    print("Hipoteza 1: normalizacja zrodel")

    # Dla normalizacji = negatywna (brak)
    x_neg = x_raw.apply(text_preprocessing_without_normalize)
    x_test_neg = x_test_raw.apply(text_preprocessing_without_normalize)
    x_train_neg, x_test_neg = vectorize(x_neg, x_test_neg)
    f1_neg = evaluate(x_train_neg, x_test_neg, y_train, y_test)

    # Dla normalizacji = positive (zawiera)
    x_pos = x_raw.apply(text_preprocessing)
    x_test_pos = x_test_raw.apply(text_preprocessing)
    x_train_pos, x_test_pos = vectorize(x_pos, x_test_pos)
    f1_pos = evaluate(x_train_pos, x_test_pos, y_train, y_test)

    print("=" * 50)
    print(f"Wynik bez normalizacji: {f1_neg*100:.2f}%")
    print(f"Wynik z normalizacja {f1_pos*100:.2f}%")

    """============== Hipoteza (H2) ================"""
    print("Hipoteza 2: Kombinacja cech TF-IDF")

    f1_tfidf = evaluate(x_train_pos, x_test_pos, y_train, y_test)

    train_sent = extract_features_batch(x_raw.tolist())
    test_sent = extract_features_batch(x_test_raw.tolist())
    x_train_combined = hstack([x_train_pos, csr_matrix(train_sent.values)])
    x_test_combined = hstack([x_test_pos, csr_matrix(test_sent.values)])
    f1_combined = evaluate(x_train_combined, x_test_combined, y_train, y_test)

    print("=" * 50)
    print(f"Wynik z sentymentem: {f1_combined*100:.2f}%")
    print(f"Wynik bez sentymentu: {f1_tfidf*100:.2f}%")

if __name__ == "__main__":
    main()
