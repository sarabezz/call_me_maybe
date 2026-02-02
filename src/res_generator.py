import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.decode import tokenize_func_names
from src.function_generation import generate_function, get_matching_functions


def save_res(prompts: list[str], functions: list[dict], model, output_path: str):
    tokenized_functions = tokenize_func_names(model, functions)

    res = []
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f"Processing: {prompt}")
            generated_fn = generate_function(model, tokenized_functions, prompt)
            if not generated_fn or "(" not in generated_fn:
                continue
            fn_name = generated_fn.split("(")[0].strip() #de "fn_add_numbers(a, b)" -> "fn_add_numbers"
            args_str = generated_fn.split("(")[1].rstrip(")").strip()
            fn_def = next((f for f in functions if f["fn_name"] == fn_name), None)
            if not fn_def:
                print(f"Function not found: {fn_name}")
                continue
    
            args = {}
            if args_str:
                arg_values = [v.strip().strip("'\"") for v in args_str.split(",")]
                for name, typ, val in zip(fn_def["args_names"], fn_def["args_types"], arg_values):
                    try:
                        if typ == "float":
                            args[name] = float(val)
                        elif typ == "int":
                            args[name] = int(float(val))
                        elif typ == "bool":
                            args[name] = val.lower() in ("true", "1", "yes")
                        else:
                            args[name] = val
                    except ValueError:
                        args[name] = val #si fcalla dejar como string
            res.append({
                "prompt": prompt,
                "fn_name": fn_name,
                "args": args
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\nResults saved in {output_path}")