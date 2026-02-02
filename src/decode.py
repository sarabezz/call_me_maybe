import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from attachments_project_ai.llm_sdk import Small_LLM_Model

#tokenizamos todos los nombres de las funciones
def tokenize_func_names(model: Small_LLM_Model, functions: list[dict]) -> dict[str, list[int]]:
    res = {}

    for fn in functions:
        name = fn["fn_name"]
        token_ids = model._tokenizer.encode(name, add_special_tokens=False)
        res[name] = token_ids
    
    return res

#nos quedamos solo con las permitidas (ej: que empiezen con fn_ generate_ids = [101])
def get_matching_functions(tokenized_functions: dict[str, list[int]], generate_ids: list[int]) -> dict[str, list[int]]:
    matches = {}

    for name, fn_ids in tokenized_functions.items():
        if fn_ids[:len(generate_ids)] == generate_ids:
            matches[name] = fn_ids

    return matches

#obtener los siguientes tokens permitidos
def get_allowed_next_tokens(matching_functions: dict[str, list[int]], generate_ids: list[int]) -> set[int]:
    allowed = set() #set y no lista porque no queremos dupl, el orden no importa, queremos "opc posibles" no secuencia

    pos = len(generate_ids) #si generate_ids = [101] pos = 0 solo vamos a mirar el token 0 de cada funcion

    for fn_ids in matching_functions.values():
        if pos < len(fn_ids): #esta func tiene mas tokens o se acabo?
            """ fn_ids = [101, 202, 303]
                pos = 1
                → fn_ids[1] = 202"""
            allowed.add(fn_ids[pos]) #allowed = {202}

    return allowed #{202, 404} en este punto solo puedo escribir add_ o is_