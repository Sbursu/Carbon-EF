import os
from typing import Any, Dict

import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to the config.yaml in the configs directory
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "config.yaml",
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
