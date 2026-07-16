import json
from typing import Dict
from pathlib import Path


class VocabularyError(Exception):
    pass


class Vocabulary:
    def __init__(self, model) -> None:
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self._load_from_model(model)

    def _load_from_model(self, model) -> None:
        try:
            vocab_path = model.get_path_to_vocabulary_json()
        except Exception as e:
            raise VocabularyError(f"Failed to get vocabulary path: {e}")
        if not Path(vocab_path).exists():
            raise VocabularyError(f"Vocabulary file not found: {vocab_path}")
        try:
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
        except json.JSONDecodeError as e:
            raise VocabularyError(f"Invalid vocabulary JSON: {e}")
        for token, token_id in vocab_data.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_token(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, f"<UNK_{token_id}>")

    def get_token_id(self, token: str) -> int:
        return self.token_to_id.get(token)

    @property
    def size(self) -> int:
        return len(self.id_to_token)