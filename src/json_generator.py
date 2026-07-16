import json
from pathlib import Path
from typing import List
from .models import FunctionCall


class JSONGenerator:
    @staticmethod
    def save_results(results: List[FunctionCall], output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_data = [result.model_dump() for result in results]
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Failed to write output file: {e}")