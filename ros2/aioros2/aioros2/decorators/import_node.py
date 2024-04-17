from typing import Any, Callable, TypeVar
from types import ModuleType
from ._decorators import RosDefinition
from ..ros_node import RosNode
from .deferrable_accessor import DeferrableAccessor

class RosImport(RosDefinition, DeferrableAccessor):
    def __init__(self, module):
        DeferrableAccessor.__init__(self, self.resolve)
        self.__module = module
    
    def resolve(self):
        """Returns an instance of the first RosNode subclass within the passed module"""
        for _, obj in self.__module.__dict__.items():
            if isinstance(obj, type) and obj is not RosNode and issubclass(obj, RosNode):
                return obj()

def import_node(l: ModuleType):
    return RosImport(l)
         