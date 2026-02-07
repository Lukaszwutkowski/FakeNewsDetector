"""
Testy dla modulu predictor
"""

import pytest

from app.predictor import predict
from utils.text_processing import text_preprocessing

@pytest.fixture
def valid_prediction_keys():
    return ["prediction", "confidence", "label"]

class TestPreprocessTextLowercase:
    """
    Testy przetwarzania tekstu na male litery
    """
    def test_it_should_return_lowercase_text_when_given_uppercase_text(self):
        # GIVEN
        text = "TEST TEXT"

        # WHEN
        result = text_preprocessing(text)

        #THEN
        assert text_preprocessing(text) == "test text"

    def test_it_should_return_lowercase_text_when_given_mixed_case_text(self):
        # GIVEN
        text = "TesT tExT"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

    def test_it_should_return_lowercase_text_when_given_lowercase_text(self):
        # GIVEN
        text = "test text"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

class TestPreprocessSpecialChars:
    """
    Testy usuwania znakow specjalnych.
    """

    def test_it_should_removes_punctuation(self):
        # GIVEN
        text = "Test, Text!!"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

    def test_it_should_removes_exclamation(self):
        # GIVEN
        text = "Test!!!! Text!!"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

    def test_it_should_removes_special_symbols(self):
        # GIVEN
        text = "Test @$$£${[][@£ Text"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

class TestPreprocessTextSpaces:
    """
    Testy usuwania i normalizacji spacji
    """

    def test_it_should_remove_multiple_spaces(self):
        # GIVEN
        text = "Test   Text"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

    def test_it_should_remove_trailing_spaces(self):
        # GIVEN
        text = "    Test Text   "

        # WHEN
        result = text_preprocessing(text)

        #THEN
        assert text_preprocessing(text) == "test text"

    def test_it_should_remove_tabs_spaces(self):
        # GIVEN
        text = "Test\tText"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test text"

class TestPreprocessTextNumbers:
    """
    Testy zachowania liczb w tekscie
    """

    def test_it_should_keep_numbers(self):
        # GIVEN
        text = "Test 123 Text"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "test 123 text"

    def test_it_should_return_only_numbers_when_given_text_with_only_numbers(self):
        # GIVEN
        text = "1234567890"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert text_preprocessing(text) == "1234567890"

class TestPreprocessTextEdgeCases:
    """
    Testy zachowania na blednych danych
    """

    def test_it_should_return_empty_string_when_given_empty_string(self):
        # GIVEN
        text = ""

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert result == ""

    def test_it_should_return_empty_string_when_given_only_spaces(self):
        # GIVEN
        text = "     "

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert result == ""

    def test_it_should_return_empty_string_when_given_none(self):
        # GIVEN
        text = None

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert result == ""

class TestPredictArticleStructure:
    """
    Testy funkcji predict
    """

    def test_it_should_returns_dictionary_when_given_text(self):
        # GIVEN
        text = "Test text"

        # WHEN
        result = predict(text)

        # THEN
        assert isinstance(result, dict)

    def test_it_should_return_prediction_key(self):
        # GIVEN
        text = "Test text"

        # WHEN
        result = predict(text)

        # THEN
        assert "prediction" in result

    def test_it_should_return_confidence_key(self):
        # GIVEN
        text = "Test text"

        # WHEN
        result = predict(text)

        # THEN
        assert "confidence" in result

    def test_it_should_return_label_key(self):
        # GIVEN
        text = "Test text"

        # WHEN
        result = predict(text)

        # THEN
        assert "label" in result

    def test_it_should_return_all_required_keys(self, valid_prediction_keys):
        # GIVEN
        text = "Test text"

        # WHEN
        result = predict(text)

        # THEN
        for key in valid_prediction_keys:
            assert key in result, f"Brak klucza {key} w wyniku"

class TestSourceCleaner:


    """
    Testy sprawdzaja czy usuwanie meta danych zrodlowych nie powoduje blednych wynikow
    """

    @pytest.mark.parametrize("input_text, expected_output", [
        ("WASHINGTON (Reuters) - The president announced new policy",
         "the president announced new policy"),
        ("New York Times (AP) - Stock markets rose today", "stock markets rose today"),
    ])
    def test_it_should_remove_city_and_news_agency_prefix(self, input_text, expected_output):
        # GIVEN --> @pytest.mark
        # WHEN
        result = text_preprocessing(input_text)

        # THEN
        assert result == expected_output

    def test_it_should_remove_news_agency_name_mention_in_text(self):
        # GIVEN
        text = "The president announced new policy according to CNN"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert "CNN" not in result

    def test_it_should_handle_text_without_source(self):
        # GIVEN
        text = "The president announced new policy"

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert result == "the president announced new policy"

    def test_it_should_return_empty_for_none(self):
        # GIVEN
        text = None

        # WHEN
        result = text_preprocessing(text)

        # THEN
        assert result == ""




