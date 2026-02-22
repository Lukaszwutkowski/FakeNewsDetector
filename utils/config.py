"""
Plik konfiguracyjny:
- wczytanie conf z pliku config.yaml
- ustalenie katalogu glownego
- sciezki absolutne do katalogow i plikow

Pozwoli to na zmiany konfiguracji bez koniecznosci zmiany kodu,
latwiej utrzymac projekt oraz zapobiega to sztywnemu wpisywnaiu sciezek na sztywno w kodzie
"""

from pathlib import Path
from typing import Any

import yaml


def _find_root():
    """
    Znajdowanie katalogu glownego projektu.
    -> Przechodzenie w gory struktury katalogow do znalezienia pliku config.yaml
    """
    current = Path(__file__).resolve().parent

    while current != current.parent:
        if (current / 'config.yaml').exists():
            return current
        current = current.parent

    raise FileNotFoundError('config.yaml nie znaleziono!')

PROJECT_ROOT = _find_root()

def load_config() -> dict[str, Any]:
    """Wczytanie konfiguracji z pliku config.yaml"""
    config_path = PROJECT_ROOT / 'config.yaml'
    return yaml.safe_load(config_path.read_text())

# Wczytanie konfiguracji przy starcie
_config = load_config()

class Paths:
    """
    Klasa zawierajaca sciezki do katalogow projektu
    Sciezki na podstawie katalogu glownego i wczytanej z config.yaml
    """
    DATA_RAW = PROJECT_ROOT / _config['paths']['data_raw']
    DATA_PROCESSED = PROJECT_ROOT / _config['paths']['data_processed']
    DATA_NEWDATASET = PROJECT_ROOT / _config['paths']['data_newDataset']
    MODELS = PROJECT_ROOT / _config['paths']['models']
    TEMPLATES = PROJECT_ROOT / _config['paths']['templates']
    STATIC = PROJECT_ROOT / _config['paths']['static']

class Files:
    """
    Klasa zawierajaca sciezki absolutne dla plikow z danymi, zapisanych modeli oraz TF-IDF
    """
    NEWS_ARTICLES = PROJECT_ROOT / _config["files"]["news_articles"]
    DATA_FAKE = PROJECT_ROOT / _config["files"]["newDataset_fake"]
    DATA_TRUE = PROJECT_ROOT / _config["files"]["newDataset_true"]

def dir_exists() -> None:
    """
    Tworzenie wymaganych katalogwo jezeli nie istnieja.
    Funkcja zostaje wywolana przy starcie aplikacji lub treningu modelu
    Pozwoli to uniknac bledu zapisu plikow
    """
    Paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    Paths.MODELS.mkdir(parents=True, exist_ok=True)
    Paths.DATA_RAW.mkdir(parents=True, exist_ok=True)
    Paths.DATA_NEWDATASET.mkdir(parents=True, exist_ok=True)

class ModelConfig:
    """
    Konfiguracja modelu. Wybor aktualnego datasetu
    """
    ACTIVE_MODEL = _config.get('model', {}).get('active_model', 'best_model')
