"""
Configuration Parser Utility
Loads YAML configuration files into convenient dict / attribute objects.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigDict(dict):
    """Dot-accessible dictionary for configuration parameters."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_yaml(file_path: str | Path) -> ConfigDict:
    """Load a single YAML file and return a ConfigDict."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ConfigDict(data)


def load_all_configs(configs_dir: str | Path = "configs") -> ConfigDict:
    """Load and merge all config files in the configs directory."""
    base_dir = Path(configs_dir)
    merged: Dict[str, Any] = {}
    for yaml_file in sorted(base_dir.glob("*.yaml")):
        key = yaml_file.stem
        merged[key] = load_yaml(yaml_file).get(key, load_yaml(yaml_file))
    return ConfigDict(merged)
