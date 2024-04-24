from typing import TypeVar
from ._decorators import RosDefinition
import dataclasses
from .deferrable_accessor import DeferrableAccessor

class RosParamReference:
    def __init__(self, params_def, param_name):
        self.params_def = params_def
        self.param_name = param_name

class RosParams(RosDefinition, DeferrableAccessor):
    def __init__(self, params_dclass) -> None:
        self.params_class = params_dclass

    # Returns array of listeners that caller can add to.
    def __getattr__(self, attr):
        field_names = [f.name for f in dataclasses.fields(self.params_class)]
        
        if not attr in field_names:
            raise AttributeError(f">{attr}< is not a field in >{self.params_class}<")

        return RosParamReference(self, attr)

T = TypeVar("T")
def params(dataclass_param: T) -> T:
    return RosParams(dataclass_param)