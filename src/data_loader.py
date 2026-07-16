import json
from pathlib import Path
from typing import List
from pydantic import ValidationError

from .models import PromptTest, FunctionDefinition


class DataLoadError(Exception):
    pass


def load_prompts(path: str) -> List[PromptTest]:
    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError(f"Prompts file not found: {path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in prompts file: {e}")
    if not isinstance(data, list):
        raise DataLoadError("Prompts file must contain a JSON list")
    try:
        return [PromptTest(**item) for item in data]
    except ValidationError as e:
        raise DataLoadError(f"Invalid prompt format: {e}")


def load_functions(path: str) -> List[FunctionDefinition]:
    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError(f"Functions file not found: {path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in functions file: {e}")
    if not isinstance(data, list):
        raise DataLoadError("Functions file must contain a JSON list")
    try:
        return [FunctionDefinition(**item) for item in data]
    except ValidationError as e:
        raise DataLoadError(f"Invalid function definitions: {e}")