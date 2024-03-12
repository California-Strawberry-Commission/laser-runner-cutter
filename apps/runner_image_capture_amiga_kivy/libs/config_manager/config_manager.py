import os
import json


class ConfigManager:
    def __init__(self, file_path, default_config={}):
        self.file_path = os.path.expanduser(file_path)
        self.data = default_config
        try:
            with open(self.file_path, "r") as file:
                self.data.update(json.load(file))
        except FileNotFoundError:
            pass

    def write_config(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as file:
            json.dump(self.data, file, indent=2)

    def has_key(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
