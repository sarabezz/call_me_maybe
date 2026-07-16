import re
from typing import List, Set, Optional, Dict, Any
from .models import FunctionDefinition
from .vocabulary import Vocabulary


class FunctionSelector:
    def __init__(self, model, vocabulary: Vocabulary,
                 functions: List[FunctionDefinition]) -> None:
        self.model = model
        self.vocab = vocabulary
        self.functions = functions
        self._tokenized_functions = self._tokenize_function_names()

    def _tokenize_function_names(self) -> Dict[str, List[int]]:
        """Tokenise each function name and store as list of token IDs."""
        tokenized = {}
        for fn in self.functions:
            tokens = self.model.encode(fn.fn_name)
            tokenized[fn.fn_name] = tokens
        return tokenized

    def select_function(self, prompt: str) -> Optional[FunctionDefinition]:
        """Select function using constrained decoding on function name tokens."""
        # Construye el contexto con descripciones para que el modelo entienda
        functions_context = "Available functions:\n"
        for fn in self.functions:
            args = ", ".join(f"{k}: {v}" for k, v in fn.args_types.items())
            functions_context += f"- {fn.fn_name}({args}) -> {fn.return_type}\n"
            if fn.description:
                functions_context += f"  Description: {fn.description}\n"

        # El prompt termina con "fn_" para que el modelo continúe con el nombre
        instruction = f"<|im_start|>system\nYou are a function calling assistant. Given a user request, you must call the most appropriate function.\n\nAvailable functions:\n{functions_context}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nfn_"
        instruction_ids = self.model.encode(instruction)

        generate_ids: List[int] = []
        max_tokens = 20

        for _ in range(max_tokens):
            # Comprueba si los tokens generados ya forman un nombre completo
            if generate_ids:
                for fn_name, fn_tokens in self._tokenized_functions.items():
                    # fn_tokens[1:] porque el primer token es "fn_" que ya está en el prompt
                    if generate_ids == fn_tokens[1:]:
                        for fn in self.functions:
                            if fn.fn_name == fn_name:
                                return fn

            # Solo permite tokens que sean continuación válida de un nombre de función
            valid_tokens = self._get_valid_function_tokens(generate_ids)
            if not valid_tokens:
                break

            full_input = instruction_ids + generate_ids
            try:
                logits = self.model.get_logits_from_input_ids(full_input)
            except Exception:
                break

            masked_logits = self._mask_logits(logits, valid_tokens)
            # DEBUG - solo para el primer step
            if len(generate_ids) <= 1:
                top = sorted(enumerate(masked_logits), key=lambda x: x[1], reverse=True)
                top = [(i, v) for i, v in top if v != float('-inf')][:10]
                print(f"  TOP TOKENS: {[(self.vocab.get_token(i), v) for i, v in top]}")
            max_logit = max(masked_logits)
            if max_logit == float('-inf'):
                break

            next_token = masked_logits.index(max_logit)
            generate_ids.append(next_token)

        return None

    def _get_valid_function_tokens(self, generated_ids: List[int]) -> Set[int]:
        """Return valid next token IDs given what has been generated so far."""
        valid: Set[int] = set()
        for fn_name, fn_tokens in self._tokenized_functions.items():
            # Quitamos el primer token que corresponde a "fn_"
            fn_tokens_without_prefix = fn_tokens[1:] if len(fn_tokens) > 1 else fn_tokens
            if len(generated_ids) < len(fn_tokens_without_prefix):
                match = all(
                    generated_ids[i] == fn_tokens_without_prefix[i]
                    for i in range(len(generated_ids))
                )
                if match:
                    valid.add(fn_tokens_without_prefix[len(generated_ids)])
        return valid

    def _mask_logits(self, logits: List[float], valid_tokens: Set[int]) -> List[float]:
        """Set all logits to -inf except valid tokens."""
        return [
            l if i in valid_tokens else float('-inf')
            for i, l in enumerate(logits)
        ]

    def extract_args(self, prompt: str, function: FunctionDefinition) -> Dict[str, Any]:
        """Extract arguments from prompt based on function definition."""
        args: Dict[str, Any] = {}
        for arg_name in function.args_names:
            arg_type = function.args_types[arg_name]
            if arg_type == "float":
                args[arg_name] = self._extract_number(prompt, args, is_float=True)
            elif arg_type == "int":
                args[arg_name] = self._extract_number(prompt, args, is_float=False)
            elif arg_type == "str":
                args[arg_name] = self._extract_string(prompt, args)
            elif arg_type == "bool":
                args[arg_name] = self._extract_bool(prompt, function, arg_name)
        return args

    def _extract_number(self, prompt: str, already_extracted: Dict,
                        is_float: bool):
        """Extract number from prompt using regex."""
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', prompt)
        index = sum(1 for v in already_extracted.values() if isinstance(v, (int, float)))
        if index < len(numbers):
            return float(numbers[index]) if is_float else int(numbers[index])
        return 0.0 if is_float else 0

    def _extract_string(self, prompt: str, already_extracted: Dict) -> str:
        """Extract string from prompt — looks for quoted values first."""
        matches = re.findall(r"'([^']*)'", prompt)
        if not matches:
            matches = re.findall(r'"([^"]*)"', prompt)
        index = sum(1 for v in already_extracted.values() if isinstance(v, str))
        if index < len(matches):
            return matches[index]
        # fallback: última palabra del prompt
        words = prompt.strip().split()
        return words[-1] if words else ""

    def _extract_bool(self, prompt: str, function: FunctionDefinition,
                      arg_name: str) -> bool:
        """Use LLM constrained to True/False to extract boolean."""
        context = f"'{prompt}'\n{function.fn_name}({arg_name}="
        input_ids = self.model.encode(context)
        true_ids = self.model.encode("True")
        false_ids = self.model.encode("False")
        if not true_ids or not false_ids:
            return True
        valid = {true_ids[0], false_ids[0]}
        logits = self.model.get_logits_from_input_ids(input_ids)
        masked = [l if i in valid else float('-inf') for i, l in enumerate(logits)]
        next_token = masked.index(max(masked))
        return next_token == true_ids[0]