import dataclasses
from typing import TypeVar


class RosParamReference:
    prefix = None
    suffix = None

    def __init__(self, params_def, param_name):
        self.params_def = params_def
        self.param_name = param_name

    def __add__(self, other):
        # Combine into new rosReference
        if isinstance(other, RosParamReference):
            pass

    def __radd__(self, other):
        pass

    def value():
        pass


class RosParams:
    def __init__(self, params_dataclass) -> None:
        self.params_dataclass = params_dataclass

    # Returns array of listeners that caller can add to.
    def __getattr__(self, attr):
        field_names = [f.name for f in dataclasses.fields(self.params_dataclass)]

        if not attr in field_names:
            raise AttributeError(
                f">{attr}< is not a field in >{self.params_dataclass}<"
            )

        return RosParamReference(self, attr)


T = TypeVar("T")


def params(params_dataclass: T) -> T:
    """
    Defines a parameters object that can be published to by the node.

    Args:
        params_dataclass: Class decorated with `@dataclass` that defines the params object.
    """
    return RosParams(params_dataclass)
