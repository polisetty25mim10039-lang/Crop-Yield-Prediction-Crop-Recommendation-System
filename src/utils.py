"""
utils.py
--------
Shared helper functions: config loading and input validation.
Keeping these in one place supports the maintainability and reliability
non-functional requirements.
"""

import os
import yaml


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or is malformed."""


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the YAML configuration file into a dict."""
    if not os.path.exists(config_path):
        raise ConfigError(f"Config file not found at '{config_path}'")
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config file: {e}")
    if not config:
        raise ConfigError("Config file is empty or invalid.")
    return config


class ValidationError(Exception):
    """Raised when user-supplied input fails validation."""


def validate_range(name: str, value: float, low: float, high: float) -> float:
    """Ensure a numeric value falls within a sane physical range."""
    if value is None:
        raise ValidationError(f"'{name}' is required.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{name}' must be a number, got: {value!r}")
    if not (low <= value <= high):
        raise ValidationError(
            f"'{name}'={value} is out of the expected range [{low}, {high}]."
        )
    return value


def validate_choice(name: str, value: str, choices) -> str:
    """Ensure a categorical value is one of the allowed choices (case-insensitive)."""
    if value is None:
        raise ValidationError(f"'{name}' is required.")
    value_norm = str(value).strip().lower()
    choices_norm = {str(c).strip().lower(): c for c in choices}
    if value_norm not in choices_norm:
        raise ValidationError(
            f"'{name}'={value!r} is not one of the accepted values: {sorted(choices)}"
        )
    return choices_norm[value_norm]


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of a file path if it doesn't already exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
