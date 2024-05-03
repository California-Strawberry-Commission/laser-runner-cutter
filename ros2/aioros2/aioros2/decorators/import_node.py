from types import ModuleType
from typing import Optional
from ._decorators import RosDefinition
from .node import RosNode
from .deferrable_accessor import DeferrableAccessor


class RosImport(RosDefinition, DeferrableAccessor):
    def __init__(
        self,
        module,
        node_name: Optional[str] = None,
        node_namespace: Optional[str] = None,
    ):
        DeferrableAccessor.__init__(self, self.resolve)
        self.__module = module
        self.node_name = node_name
        self.node_namespace = node_namespace

    def resolve(self):
        """Returns an instance of the first RosNode subclass within the passed module"""
        for _, obj in self.__module.__dict__.items():
            if (
                isinstance(obj, type)
                and obj is not RosNode
                and issubclass(obj, RosNode)
            ):
                return obj()


def import_node(
    module: ModuleType,
    node_name: Optional[str] = None,
    node_namespace: Optional[str] = None,
):
    return RosImport(module, node_name, node_namespace)
