"""
Skrypt zawierajacy slownik modeli oraz metody pozwalajace na korzystanie z
konlretnego modelu
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

def all_models():
    """
    Slownik z kilkoma modelami. Zaimplementowane modele:
    - Logistic Regression:
        Klasyfikacja binarna, jego zadaniem jest przwidywanie prawdobodobienstwa
        przynaleznosci do jednej z dwoch klas. W przypadku tego projektu jest to
        klasa 1 (Fake jesli P > 0.5) lub 0 (True P <= 0.5)
    - Random Forest:
        Zespol drzew decyzyjnych, klasyfikacja wieloklasowa.
        Tworzenie N drzew decyzyjnych, gdzie treningowe zbiory sa przetworzone przez kazde drzewo.
        Wybor klasy poprzez decyzje wielu drzew (wiekszosciowa).
    - LightGBM :
        Model budowany sekwencyjnie, nauka modeli na bledach poprzedniego modelu
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            class_weight='balanced',
            solver='lbfgs'
        ),

        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            max_depth=50,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1 # --> uzycie wszystkich dostepnych rdzeni procesora
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            num_leaves=31,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1 # wylaczenie komunikatow
        )
    }

    return models


def get_model(name: str):
    """
    Funkcja zwraca wybrany model
    """
    models = all_models()
    return models.get(name)

def get_available_models():
    """
    Lista z dostepnymi modelami
    """
    return list(all_models().keys())