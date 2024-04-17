from typing import TypeVar
from ._decorators import RosDefinition

class RosParams(RosDefinition):
    def __init__(self, params_dclass) -> None:
        self.params_class = params_dclass

T = TypeVar("T")
def params(dataclass_param: T) -> T:
    return RosParams(dataclass_param)