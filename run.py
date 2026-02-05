"""
Skrypt startowy, uruchomienie aplikacji (start serwera Flask)
"""

from app.app import app

if __name__ == '__main__':
    print("=" * 10 + "START APPLICATION" + "=" * 10)
    print("Serwer uruchomiony na porcie 5000")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )