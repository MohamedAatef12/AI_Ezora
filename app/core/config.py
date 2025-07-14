import yaml
import os

class AppConfig:
    def __init__(self, config_path="app\core\config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, section: str, key: str):
        return self.config.get(section, {}).get(key)

    def all(self):
        return self.config
