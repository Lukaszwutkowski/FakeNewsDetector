"""
Modul odpowiedzialny za predykcje. Laduje wytrenowany model i klasyfikuje tekst
"""

import joblib

from utils.text_processing import text_preprocessing
from utils.config import Files

model = None
vectorizer = None

def load_model():
    """
    Ladowanie modelu oraz wektoryzatora z plikow
    """
    global model, vectorizer

    if Files.MODEL.exists() and Files.VECTORIZER.exists():
        model = joblib.load(Files.MODEL)
        vectorizer = joblib.load(Files.VECTORIZER)
        print("Model zaladowany pomyslnie")
    else:
        print("Nie znaleziono modelu")
        model = None
        vectorizer = None

def predict(text):
    """
    Predykcja dla podanego tekstu, przyjmuje parametr text (str) czyli tekst artykulu
    Zwraca: Slownik z wynikami predykcji:
    - prediction: 0 - Fake News, 1 - Real News
    - confidence: poziom prawidlowosci predykcji (0-100)
    - label: etykieta predykcji
    """
    global model, vectorizer

    if model is None or vectorizer is None:
        load_model()

    if model is None or vectorizer is None:
        return {
            "prediction": -1,
            "confidence": 0,
            "label": "Nieznany"
        }

    processed_text = text_preprocessing(text)

    text_vectorized = vectorizer.transform([processed_text])

    prediction = model.predict(text_vectorized)[0]

    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities) * 100

    if prediction == 0:
        label = "Fake News"
    else:
        label = "Real News"

    return {
        "prediction": int(prediction),
        "confidence": round(confidence, 1),
        "label": label
    }

load_model()

if __name__ == "__main__":
    test = "This news is fake. It is not true. It is a lie."
    print(predict(test))

