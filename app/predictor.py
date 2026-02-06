"""
Modul odpowiedzialny za predykcje. Laduje wytrenowany model i klasyfikuje tekst
"""

import joblib

from utils.text_processing import text_preprocessing
from utils.config import Paths

RUN_NAME = "new_dataset"
MODEL_PATH = Paths.MODELS / RUN_NAME / "model.joblib"
VECTORIZER_PATH = Paths.MODELS / RUN_NAME / "vectorizer.joblib"

model = None
vectorizer = None

def load_model():
    """
    Ladowanie modelu oraz wektoryzatora z plikow
    """
    global model, vectorizer

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model zaladowany pomyslnie")
    else:
        print("Nie znaleziono modelu")
        model = None
        vectorizer = None

def predict(text):
    """
    Predykcja dla podanego tekstu, przyjmuje parametr text (str) czyli tekst artykulu
    Zwraca: Slownik z wynikami predykcji:
    - prediction: 1 - Fake News, 0 - Real News
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

    prediction = int(model.predict(text_vectorized)[0])

    probabilities = model.predict_proba(text_vectorized)[0]
    real_probability = probabilities[0] * 100
    fake_probability = probabilities[1] * 100

    label = "Fake News" if prediction == 1 else "Real News"
    confidence = fake_probability if prediction == 1 else real_probability

    return {
        "prediction": int(prediction),
        "confidence": round(confidence, 1),
        "fake_probability": round(fake_probability, 1),
        "label": label
    }

load_model()

if __name__ == "__main__":
    test = "This news is fake. It is not true. It is a lie."
    print(predict(test))

