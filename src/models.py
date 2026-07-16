from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class ParameterType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, ParameterType]
    returns: ParameterType

    @property
    def fn_name(self) -> str:
        return self.name

    @property
    def args_names(self) -> List[str]:
        return list(self.parameters.keys())

    @property
    def args_types(self) -> Dict[str, str]:
        type_map = {
            "number": "float",
            "string": "str",
            "boolean": "bool",
            "integer": "int"
        }
        return {k: type_map.get(v.type, v.type) for k, v in self.parameters.items()}

    @property
    def return_type(self) -> str:
        type_map = {
            "number": "float",
            "string": "str",
            "boolean": "bool",
            "integer": "int"
        }
        return type_map.get(self.returns.type, self.returns.type)


class PromptTest(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    fn_name: str
    args: Dict[str, Any]