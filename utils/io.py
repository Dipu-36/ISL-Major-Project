import csv
import json
import os
from typing import Any, Dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_uid_to_text(csv_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = row.get("uid")
            text = row.get("text")
            if uid is not None and text is not None:
                mapping[uid] = text
    return mapping
