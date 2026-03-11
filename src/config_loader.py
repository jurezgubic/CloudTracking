"""Load and validate TOML configuration files for CloudTracker."""

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Keys that stay nested (not flattened)
_NESTED_KEYS = {"file_name"}

# Required keys per data format
_REQUIRED_KEYS_COMMON = {"data_format"}
_REQUIRED_KEYS_UCLA = {"base_file_path", "file_name"}
_REQUIRED_KEYS_MONC = {"monc_data_path", "monc_config_file"}


def load_config(path: str) -> dict[str, Any]:
    """Load a TOML config file and return a flat dict.

    TOML sections are flattened into a single-level dict so the result
    is identical in structure to the former inline config dict in main.py.
    The ``file_name`` sub-table is preserved as a nested dict.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    config = _flatten(raw)
    validate_config(config)
    return config


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested TOML sections into a single-level dict."""
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and key not in _NESTED_KEYS:
            # Recurse into sections, but preserve _NESTED_KEYS as-is
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, dict) and inner_key in _NESTED_KEYS:
                    flat[inner_key] = inner_value
                elif isinstance(inner_value, dict) and inner_key not in _NESTED_KEYS:
                    # Two levels deep (shouldn't happen, but handle gracefully)
                    for k, v in inner_value.items():
                        flat[k] = v
                else:
                    flat[inner_key] = inner_value
        else:
            flat[key] = value
    return flat


def validate_config(config: dict[str, Any]) -> None:
    """Check that required keys are present and data paths exist."""
    missing = _REQUIRED_KEYS_COMMON - config.keys()
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    data_format = config["data_format"]

    if data_format == "UCLA-LES":
        missing = _REQUIRED_KEYS_UCLA - config.keys()
        if missing:
            raise ValueError(f"UCLA-LES format requires keys: {missing}")
        _check_path_exists(config["base_file_path"], "base_file_path")

    elif data_format == "MONC":
        missing = _REQUIRED_KEYS_MONC - config.keys()
        if missing:
            raise ValueError(f"MONC format requires keys: {missing}")
        _check_path_exists(config["monc_data_path"], "monc_data_path")
        _check_path_exists(config["monc_config_file"], "monc_config_file")

    else:
        raise ValueError(f"Unknown data_format '{data_format}'. Expected 'UCLA-LES' or 'MONC'.")


def _check_path_exists(path: str, key_name: str) -> None:
    """Raise FileNotFoundError if path does not exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Path for '{key_name}' does not exist: {path}")
