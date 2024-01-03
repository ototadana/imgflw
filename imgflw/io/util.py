import os

import requests

assets_dir = os.path.abspath(os.environ.get("IMGFLW_ASSETS_DIR", None) or "./assets")
module_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"module_dir={module_dir}", flush=True)
print(f"assets_dir={assets_dir}", flush=True)


def get_asset(path: str) -> str:
    return os.path.join(assets_dir, path)


def get_module(path: str) -> str:
    return os.path.join(module_dir, path)


def fetch(url: str, **kwargs) -> str:
    response = requests.get(url, **kwargs)
    return response.text
