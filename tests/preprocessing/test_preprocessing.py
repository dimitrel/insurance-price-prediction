import pytest
from insurance_price_prediction.preprocessing.text_preprocessing import (
    clean_text,
    preprocess_text_with_nltk,
)
import nltk
nltk.download('punkt')

@pytest.mark.parametrize("input_text, keep_numbers, expected_output", [
    ("Hello World!!", False, "hello world"),
    ("123Hello 456World", False, "hello world"),
    ("Product 2024 Model-X", True, "product 2024 model x"),
    ("NoSpecialCharsHere", False, "nospecialcharshere"),
    ("      Extra  Spaces    ", False, "extra spaces"),
    ("123456789", False, ""),
    ("Test12345String", True, "test12345string")
])
def test_clean_text(input_text, keep_numbers, expected_output):
    assert clean_text(input_text, keep_numbers) == expected_output


@pytest.mark.parametrize("input_text, lemmatizer, expected_output", [
    ("Running quickly to the store", True, "running quickly store"),
    ("The dogs are barking loudly", True, "dog barking loudly"),
    ("Testing functions is important", True, "testing function important"),
    ("A big red apple", True, "big red apple"),
    ("An apple a day keeps the doctor away", True, "apple day keep doctor away"),
    ("This is a simple test", True, "simple test")
])
def test_preprocess_text_with_nltk(input_text, lemmatizer, expected_output):
    assert preprocess_text_with_nltk(input_text, lemmatizer) == expected_output
