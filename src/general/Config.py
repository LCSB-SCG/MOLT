from collections.abc import MutableMapping
import json


class Config(MutableMapping):
    def __init__(self, config_path: dict) -> None:
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            self.store = json.loads(f.read())

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __str__(self) -> str:
        return str(json.dumps(self.store, indent=4))

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key
