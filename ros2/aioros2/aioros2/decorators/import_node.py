from typing import Callable, TypeVar
from ._decorators import RosDefinition

class RosImport(RosDefinition):
    def __init__(self, instance_lambda):
        self.__get_instance = instance_lambda
    
    def resolve(self):
        return self.__get_instance()

T = TypeVar("T")
def import_node(l: Callable[[], T]) -> T:
    return RosImport(l)
         