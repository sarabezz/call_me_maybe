import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))
from attachments_project_ai.llm_sdk import Small_LLM_Model
from src.decode import get_matching_functions, get_allowed_next_tokens


def mask_logits(logits, allowed_tokens):
    masked = [-float("inf")] * len(logits)
    for t in allowed_tokens:
        if t < len(masked):
            masked[t] = logits[t]

    return masked

def get_next_token(model, full_input_ids, generated_ids, matching_functions):
    allowed_tokens = get_allowed_next_tokens(matching_functions, generated_ids)
    if not allowed_tokens:
        return None
    try:
        logits = model.get_logits_from_input_ids(full_input_ids)
    except Exception as e:
        print(f"Error obteniendo logits: {e}")
        return None
    masked_logits = mask_logits(logits, allowed_tokens)
    max_logit = max(masked_logits)
    if max_logit == -float("inf"): #verificar si al menos hay 1 token valido
        return None

    return masked_logits.index(max_logit)
    
#genera una funcion valida usando decod restringida, sin posibilidad de inventarse nada
def generate_function(model, tokenized_functions, prompt) -> str:
    prompt_ids = model._encode(prompt).tolist()[0]
    generated_ids = []

    with torch.no_grad(): #no caclula gradientes(entreamiento)
        for step in range(20): #escribir la funcion token a token, bucle hasta generar la funcion completa
            matching_functions = get_matching_functions(tokenized_functions, generated_ids) #filtrar funciones que van coincidiendo con lo que vamos generando
            if not matching_functions:
                break
            
            full_input = prompt_ids + generated_ids
            next_token = get_next_token(model, full_input, generated_ids, matching_functions)
            if next_token is None:
                break
            
            generated_ids.append(next_token) #añadir tokens a la lista
        for step in range(30):
            full_input = prompt_ids + generated_ids
            try:
                logits = model.get_logits_from_input_ids(full_input)
            except Exception as e:
                print(f"Error: {e}")
                break
            next_token = logits.index(max(logits))
            generated_ids.append(next_token)
            
            token_str = model._tokenizer.decode([next_token])
            if ")" in token_str:
                break

    if not generated_ids:
        return ""
    
    res = model._tokenizer.decode(generated_ids).strip() #decodificar resultado final
    if ")" in res:
        res = res[:res.index(")") + 1]
    print("Función generada:", res)
    return res