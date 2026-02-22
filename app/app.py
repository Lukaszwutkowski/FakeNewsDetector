"""
Aplikacja Flask - routing i obsluga zadan HTTP
"""
import os

from flask import Flask, render_template, request
from app.predictor import predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inicjalizacja
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder = os.path.join(BASE_DIR, 'static')
)

@app.route('/')
def index():
    """
    Strona glowna aplikacji
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Obsluga predykcji
    """
    # 1. Pobranie danych z formularza
    title = request.form.get('title', '')
    text = request.form.get('text', '')

    # 2. Polaczenie tytulu oraz tekstu
    complete_text = f"{title}{text}".strip()

    # 3. Sprawdzenie czy tekst nie jest pusty
    if not complete_text:
        return render_template('index.html', error="Wpisz tekst do przetworzenia!")

    # 4. Predykcja
    result = predict(complete_text)

    # 5. Wyswietlenie wyniku
    return render_template('result.html',
                           title=title,
                           text=text,
                           prediction=result['prediction'],
                           confidence=result['confidence'],
                           label=result['label'])

@app.route('/about')
def about():
    """
    Strona z informacja o aplikacji
    """
    return render_template('about.html')

# print(app.template_folder)  # -> test sciezki do templates