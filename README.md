*Este proyecto ha sido creado como parte del currículo de 42 por sarabezz.*

# CALL_ME_MAYBE

A Python program that translates natural language requests into structured function calls using constrained decoding.


## About

Call Me Maybe is a function calling system that translates natural language requests into structured function calls. Given a prompt in natural language, the system uses constrained decoding to guide a small LLM token by token, restricting its output at each step to only valid function names and argument values. This guarantees a valid, parseable JSON output even with a model as small as 0.6B parameters.

