import json
import yaml
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(path):
    extension = os.path.splitext(path)[1].lower()
    if extension in [".yaml", ".yml"]:
        return load_yaml(path)
    elif extension == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
