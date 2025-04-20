import pytest
from src.preprocessing import clean_text

def test_clean_text():
    """
    Test if the function removes special characters,
    converts to lowercase, removes stopwords, and lemmatizes.
    """
    original = "I absolutely love this product! It's amazing!!!"
    expected = "absolutely love product amazing recommend"
    result = clean_text(original)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_clean_empty_text():
    """
    Test if empty text is handled.
    """
    original = ""
    expected = ""
    result = clean_text(original)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_stopwords_removal():
    """
    Test that stopwords are properly removed.
    """
    original = "This is a simple test sentence."
    expected = "simple test sentence"
    result = clean_text(original)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_special_characters_removal():
    """
    Test that special characters are removed.
    """
    original = "Hello!!! Is this a test, or just a joke?"
    expected = "hello test joke"
    result = clean_text(original)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_lowercasing():
    """
    Test that all text is converted to lowercase.
    """
    original = "HeLLo WoRLd"
    expected = "hello world"
    result = clean_text(original)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_numbers_and_punctuation():
    """
    Test that numbers and punctuation are removed.
    """
    original = "Product #1 costs $99.99! Buy it today!"
    expected = "product cost buy today"
    result = clean_text(original)
    assert result == expected, f"Expected '{expected}', but got '{result}'"
