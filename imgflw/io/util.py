import os
from typing import List

import requests

assets_dir = os.path.abspath(os.environ.get("IMGFLW_ASSETS_DIR", None) or "./assets")
module_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"module_dir={module_dir}", flush=True)
print(f"assets_dir={assets_dir}", flush=True)


def get_asset(path: str) -> str:
    return os.path.join(assets_dir, path)


def has_asset(path: str) -> bool:
    return os.path.exists(get_asset(path))


def save_asset(path: str, content: str) -> None:
    with open(get_asset(path), "w") as f:
        f.write(content)


def load_asset(path: str) -> str:
    with open(get_asset(path), "r") as f:
        return f.read()


def get_assets(path: str) -> List[str]:
    path = os.path.join(assets_dir, path)
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getatime(os.path.join(path, x)))
    return [os.path.basename(file) for file in files]


def get_module(path: str) -> str:
    return os.path.join(module_dir, path)


def fetch(url: str, **kwargs) -> str:
    response = requests.get(url, **kwargs)
    return response.text
