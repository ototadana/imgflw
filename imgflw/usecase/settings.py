import json
import os
import shutil
import tempfile
from typing import Any, Dict

import torch


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Settings:
    settings: Dict[str, Any] = {}
    path: str = "imgflw_settings.json"

    @classmethod
    def load(cls, path: str = "imgflw_settings.json"):
        cls.path = path
        cls.settings = {}
        if os.path.exists(cls.path):
            with open(cls.path, "r") as f:
                cls.settings = json.load(f)

    @classmethod
    def save(cls):
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            json.dump(cls.settings, tf, indent=4)
            temp_name = tf.name

        try:
            shutil.move(temp_name, cls.path)
        except Exception:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise

    @classmethod
    def get(cls, key: str, default: Any = "") -> Any:
        return cls.settings.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any):
        cls.settings[key] = value

    @classmethod
    def update(cls, settings: Dict[str, Any]):
        cls.settings = settings

    @classproperty
    def device(cls) -> str:
        device = cls.get("device", None)
        if device is None:
            device = os.environ.get("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            cls.set("device", device)
        return torch.device(device)
