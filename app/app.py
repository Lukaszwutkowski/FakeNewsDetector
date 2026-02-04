"""
Aplikacja Flask - routing i obsluga zadan HTTP
"""

from flask import Flask, render_template
from app.predictor import predict

# Inicjalizacja
app = Flask(__name__)

@app.route('/')
def index():
    """
    Strona glowna aplikacji
    """
    return render_template('index.html')