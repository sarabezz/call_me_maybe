import json
import sys
from pathlib import Path #manejar rutas de forma más segura y clara

sys.path.append(str(Path(__file__).resolve().parent.parent))
from attachments_project_ai.llm_sdk import Small_LLM_Model
from src.res_generator import save_res

def load_prompts(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{path} no exists")

    with open(file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list")
    prompts = []
    for case in data:
        if not isinstance(case, dict) or "prompt" not in case:
            raise ValueError(f"Invalid test case format: {case}")
        prompts.append(case["prompt"])

    return prompts

def load_functions(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{path} no extists")

    with open(file_path, "r") as f:
        functions = json.load(f)

    if not isinstance(functions, list):
        raise ValueError("functions_definition.json must be a list")
    required_keys = {"fn_name", "args_names", "args_types", "return_type"}
    for func in functions:
        if not isinstance(func, dict):
            raise ValueError("Each functions must be an object")
        if not required_keys.issubset(func.keys()):
            raise ValueError(f"Keys missing in the function definition: {func}")

    return functions

def main():
    try:
        print("Loading entry data...")
        prompts = load_prompts("attachments_project_ai/data/input/function_calling_tests.json")
        functions = load_functions("attachments_project_ai/data/input/functions_definition.json")
        print(f"{len(prompts)} prompts and {len(functions)} functions were load\n")

        #configurar el LLM
        print("Initilazing model")
        model = Small_LLM_Model()
        print(f"{model._model_name} model load in {model._device}\n")

        output_path = "attachments_project_ai/data/output/function_calling_results.json"  # ✅ Ruta completa
        print("Starting generation...")
        save_res(prompts, functions, model, output_path)

    except FileNotFoundError as e:
        print(f"\nError de archivo: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError de formato: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()