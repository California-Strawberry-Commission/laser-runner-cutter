from typing import TypeVar
from ._decorators import RosDefinition
import dataclasses
from .deferrable_accessor import DeferrableAccessor


class RosParamReference:
    prefix = None
    suffix = None

    def __init__(self, params_def, param_name):
        self._params_def = params_def
        self._param_name = param_name

    # https://mathspp.com/blog/pydonts/dunder-methods
    def __add__(self, other):
        # Combine into new rosReference
        if isinstance(other, RosParamReference):
            pass

    def __radd__(self, other):
        pass

    def value():
        pass


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
