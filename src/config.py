import yaml

_DEFAULT_CONFIG_PATH = "config/pipeline_config.yaml"


def load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
