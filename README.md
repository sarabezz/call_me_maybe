*Este proyecto ha sido creado como parte del currículo de 42 por sarabezz.*

# CALL_ME_MAYBE

A Python program that translates natural language requests into structured function calls using constrained decoding.


## About

Call Me Maybe is a function calling system that translates natural language requests into structured function calls. Given a prompt in natural language, the system uses constrained decoding to guide a small LLM token by token, restricting its output at each step to only valid function names and argument values. This guarantees a valid, parseable JSON output even with a model as small as 0.6B parameters.


## Algorithm
### Step 1 - Function selection
The model is given a prompt with the list of available functions and their descriptions. It then generates the function name token by token, but at each step only tokens that are valid continuations of a known function name are allowed. All other tokens are masked to `-inf` — making their probability mathematically zero. This guarantees the model always produces a real function name.

### Step 2 - Argument extraction
Once the function is selected, its parameter types are known from the function definition. Arguments are extracted from the prompt using regex, matching each parameter type — numbers for `float` and `int`, quoted strings for `str`, and constrained decoding between `True` and `False` for `bool`.

The constrained decoding is implemented in `_mask_logits` — it takes the full logits vector (151,643 values for Qwen3-0.6B) and sets all positions to `-inf` except the valid tokens for the current step. The model then picks the highest remaining logit, which is always a valid token.


## Desing Decisions
### Public SDK methods only
The `llm_sdk`packages provides private methods `_encode` and `_decode` but the project rules prohibit using private attributes. Two public methods `encode` and `decode` were added to the SDK wrapper to tokenise text and convert token IDs back to text without accesing private internals.

### Token ID comparison instead of string comparison
When checking if the generated tokens match a function name, the system compares lists of token IDs directly instead of decodign them back to text. This avoids issues with how the tokeniser splits subwords -- for example `fn_reverse_string` tokenises as `['fn', '_reverse', '_string'']`, so decodigns the generated tokens and prepending `fn_` would produce `fn__reverser_string` with a double underscore.

### Pydantic for dat validation
All input data -- prompts and fucntion definitions -- is validated using Pydantic models. This ensures the program handles malformed JSON, missing fields, and wrongs types gracefully instead of crashing.

### Generic argument extraction
The `ArgumetExtractor` is driven entirely by the `args_types` fiels of each function definition. It does not hardcode logic per function name -- if a new function is added to the definitions file, argument extraction works automatically based on its parameter types.

### Qwen chat format
The prompt uses Qwen3's native chat format (`<|im_start|>`, `<|im_end|>`) since the model was trained with this structure. This significantly improves function selection accuracy compared to plain text prompts.


## Performance
The system was tested against 14 prompts covering all 7 available functions.
| Metric | Result |
|--------|--------|
| Function selection accuracy | 12/14 (85%) |
| Valid JSON output | 14/14 (100%) |
| Argument extraction accuracy | 12/14 (85%) |

The 2 prompts that fail involve `fn_substitute_string_with_regex` -- prompts like `"Substitute the digits in the string..."` and `"Replace all vowels in..."`. The model assigns a much higher logit to `_reverse` than to `_sub` when the `prompt contains the word "string", regardless of the context. This is a bias in the model's training data, not a bug in the constrained decoding implementation.

With a larger model (Qwen3-7B or above) these cases would likely resolve correctly, since the model would have enough capacity to distinguish between "reverse a string" and "substitute within a string". The constrained decoding mechanism itself is correct — it always produces a valid function name and valid JSON, which is the primary goal of the technique.


## Challenges
### Token prefix mismatch
The first major bug was that comparing the generated text with function names always failed. The tokeniser splits fn_reverse_string as `['fn', '_reverse', '_string']` — so the second token is `_reverse`, not `reverse`. When decoding the generated tokens and prepending `fn_`, the resulty `fn__reverse_string` with a double underscore. The fix was to compare token ID lists directly instead of reconstructing the string.

### Private SDK methods
The original SDK only exposed `_encode` and `_decode` as private methods, but the project rules prohibit using private attributes. This meant we could not tokenise text through the official interface. The fix was to add public `encode` and `decode` methods to the SDK wrapper that call the underlying tokeniser directly.

### Function selection bias
The model consistently preferred `fn_reverse_string` for most prompts regardless of context, because `_reverse` has a much higher base logit than other function prefixes. This was partially solved by using Qwen3's native chat format (`<|im_start|>`, `<|im_end|>`), which improved accuracy from 9/14 to 12/14. The remaining 2 failures are caused by the model's training bias toward `_reverse` when the prompt contains the word "string".

### Number extraction with constrained decoding
An early implementation tried to extract numbers using the LLM with constrained decoding — only allowing digit tokens at each step. This produced values like `2000000000.0` instead of `2.0` because the model kept generating zeros with no stopping condition. The fix was to extract numbers directly from the prompt using regex, which is both simpler and more reliable since the numbers are always present in the original text.

### Memory limitations
Loading Qwen3-0.6B requires approximately 1.5GB of RAM. With WSL2's default memory limit of 3.7GB and the system already using 1.3GB, the model load caused the process to crash. The fix was to increase WSL2's memory limit to 5GB by editing `.wslconfig`.


## Testing

The implementation was validated by running the system against the 14 provided prompts and inspecting the output JSON manually.
To verify function selection, debug logging was added to print the top candidate tokens and their logits at each step of the constrained decoding process. This made it possible to see exactly why the model preferred one function over another and to measure the margin between candidates.
To verify argument extraction, the output JSON was checked against the expected values for each prompt — numbers matched the values in the original text, strings matched the quoted substrings, and booleans reflected the correct answer.
The output file was also validated structurally — every entry contains `prompt`, `fn_name`, and `args` with the correct types as defined in `functions_definition.json`, and the file is always valid JSON regardless of which function was selected.


## Usage

```bash
git clone https://github.com/sarabezz/call_me_maybe.git
cd call_me_maybe
uv sync
```

Run the program:

```bash
uv run python -m src
```

Or with custom input/output files:

```bash
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

The input directory must contain two files:
data/input/function_calling_tests.json    # list of natural language prompts
data/input/functions_definition.json      # list of function definitions

Example prompt file:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Example output:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "fn_name": "fn_reverse_string",
    "args": {"s": "hello"}
  }
]
```


## Resources
## Resources

- [Hugging Face — Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — model used for constrained decoding
- [Pydantic documentation](https://docs.pydantic.dev/) — data validation library used for input parsing
- [Constrained decoding explained — Willard & Louf (2023)](https://arxiv.org/abs/2307.09702) — the academic paper behind the technique
- [Outlines library](https://github.com/outlines-dev/outlines) — reference implementation of constrained decoding (use of the library itself is prohibited by the project rules)
- [Qwen3 technical report](https://huggingface.co/papers/2505.09388) — model architecture and training details

### AI usage

Claude (Anthropic) was used throughout this project for:
- Debugging token comparison logic and identifying the double underscore bug
- Understanding why constrained decoding was producing incorrect numbers and finding the regex solution
- Improving function selection accuracy by experimenting with different prompt formats including Qwen3's native chat format
- Drafting and iterating on this README


## Project Info
| | |
|---|---|
| Language | Python |
| School | 42 Madrid |
| Status | ✅ Completed |
