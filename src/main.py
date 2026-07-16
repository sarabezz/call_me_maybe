import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attachments_project_ai.llm_sdk import Small_LLM_Model

from .data_loader import load_prompts, load_functions, DataLoadError
from .vocabulary import Vocabulary, VocabularyError
from .function_selector import FunctionSelector
from .json_generator import JSONGenerator
from .models import FunctionCall


def main() -> int:
    try:
        input_dir = Path("attachments_project_ai/data/input")
        output_dir = Path("attachments_project_ai/data/output")

        prompts_file = input_dir / "function_calling_tests.json"
        functions_file = input_dir / "functions_definition.json"
        output_file = output_dir / "function_calling_results.json"

        # PASO 1: CARGAR DATOS
        print("📥 Loading input data...")
        try:
            prompts = load_prompts(str(prompts_file))
            functions = load_functions(str(functions_file))
            print(f"  ✅ Loaded {len(prompts)} prompts")
            print(f"  ✅ Loaded {len(functions)} function definitions\n")
        except DataLoadError as e:
            print(f"❌ Error loading data: {e}")
            return 1

        # PASO 2: INICIALIZAR LLM
        print("🤖 Initializing LLM...")
        try:
            model = Small_LLM_Model()
            print(f"  ✅ Model loaded: {model._model_name}")
            print(f"  ✅ Device: {model._device}\n")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return 1

        # PASO 3: CARGAR VOCABULARIO
        print("📖 Loading vocabulary...")
        try:
            vocabulary = Vocabulary(model)
            print(f"  ✅ Vocabulary size: {vocabulary.size} tokens\n")
        except VocabularyError as e:
            print(f"❌ Error loading vocabulary: {e}")
            return 1

        # PASO 4: INICIALIZAR SELECTOR
        print("🔧 Initializing components...\n")
        selector = FunctionSelector(model, vocabulary, functions)

        # PASO 5: PROCESAR CADA PROMPT
        results = []
        print("🚀 Processing prompts...\n")
        print("=" * 70)

        for i, prompt_obj in enumerate(prompts, 1):
            prompt = prompt_obj.prompt
            print(f"\n[{i}/{len(prompts)}] {prompt}")
            print("-" * 70)

            try:
                # Seleccionar función con constrained decoding
                function = selector.select_function(prompt)

                if function is None:
                    print("  ⚠️  Could not determine appropriate function")
                    continue

                print(f"  📌 Selected: {function.fn_name}")

                # Extraer argumentos
                args = selector.extract_args(prompt, function)
                print(f"  📌 Arguments: {args}")

                result = FunctionCall(
                    prompt=prompt,
                    fn_name=function.fn_name,
                    args=args
                )
                results.append(result)
                print("  ✅ Success")

            except Exception as e:
                print(f"  ❌ Error processing prompt: {e}")
                continue

        # PASO 6: GUARDAR RESULTADOS
        print("\n" + "=" * 70)
        print(f"\n💾 Saving results...")

        try:
            JSONGenerator.save_results(results, str(output_file))
            print(f"  ✅ Results saved to: {output_file}")
            print(f"  ✅ Total successful: {len(results)}/{len(prompts)}")
        except IOError as e:
            print(f"❌ Error saving results: {e}")
            return 1

        print("\n✨ Done!\n")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())