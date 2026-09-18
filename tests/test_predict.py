import pytest

from src.utils import validate_range, validate_choice, ValidationError


def test_validate_range_accepts_valid_value():
    assert validate_range("rainfall", 200, 0, 5000) == 200.0


def test_validate_range_rejects_out_of_range():
    with pytest.raises(ValidationError):
        validate_range("rainfall", 9000, 0, 5000)


def test_validate_range_rejects_non_numeric():
    with pytest.raises(ValidationError):
        validate_range("ph", "not-a-number", 0, 14)


def test_validate_range_rejects_none():
    with pytest.raises(ValidationError):
        validate_range("humidity", None, 0, 100)


def test_validate_choice_accepts_case_insensitive():
    assert validate_choice("season", "KHARIF", ["Kharif", "Rabi"]) == "Kharif"


def test_validate_choice_rejects_invalid_choice():
    with pytest.raises(ValidationError):
        validate_choice("season", "monsoon", ["Kharif", "Rabi"])


def test_validate_choice_rejects_none():
    with pytest.raises(ValidationError):
        validate_choice("state", None, ["Punjab", "Haryana"])
