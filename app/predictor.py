"""
Modul odpowiedzialny za predykcje. Laduje wytrenowany model i klasyfikuje tekst

Wersja 1.0:
Modul laduje wytrenowany model ML i wykonuje klasyfikacje tekstow
Uzywany przez aplikacje Flask do obslugi z uzytkownikiem
Inzynieria cech - TF-IDF na tekscie po obrobce -> Preprocessing

Wersja 1.1
Inzynieria cech - Dodatkowo analiza sentymentu (VADER)

Aktualny pipeline:
1. uzytkownik wkleja tytul i tekst
2. tekst jest poddany czyszczeniu i normalizacji (preprocessing)
3. TF-IDF zamienia tekst na wektor numeryczny (wektoryzacja)
4. VADER analizuje sentyment tekstu
5. Model ML klasyfikuje jako 1 lub 0 (Fake, Real)
6. Uzytkownik otrzymuje wynik
"""

from src.models_training import load_trained_model
from utils.feature_engineering import combine_features
from utils.text_processing import text_preprocessing
from utils.config import Paths, ModelConfig

RUN_NAME = ModelConfig.ACTIVE_MODEL


model = None
vectorizer = None
feature_extractor = None

def load_model():
    """
    Ladowanie modelu oraz wektoryzatora z plikow
    """
    global model, vectorizer, feature_extractor
    model, vectorizer, feature_extractor = load_trained_model()

def predict(text):
    """
    Predykcja dla podanego tekstu, przyjmuje parametr text (str) czyli tekst artykulu
    Zwraca: Slownik z wynikami predykcji:
    - prediction: 1 - Fake News, 0 - Real News
    - confidence: poziom prawidlowosci predykcji (0-100)
    - label: etykieta predykcji
    """
    global model, vectorizer, feature_extractor

    if model is None or vectorizer is None or feature_extractor is None:
        load_model()

    processed_text = text_preprocessing(text)

    text_vectorized = vectorizer.transform([processed_text])
    sentiment_features = feature_extractor.transform([processed_text])
    final_features = combine_features(text_vectorized, sentiment_features)

    prediction = int(model.predict(final_features)[0])

    probabilities = model.predict_proba(final_features)[0]
    real_probability = probabilities[0] * 100
    fake_probability = probabilities[1] * 100

    label = "Fake News" if prediction == 1 else "Real News"
    confidence = fake_probability if prediction == 1 else real_probability

    return {
        "prediction": int(prediction),
        "confidence": round(confidence, 1),
        "label": label
    }

load_model()

if __name__ == "__main__":
    test = "This news is fake. It is not true. It is a lie."
    print(predict(test))

