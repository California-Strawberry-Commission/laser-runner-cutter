from types import ModuleType
from typing import Optional

from .node import RosNode
from .lazy_accessor import LazyAccessor


class RosImport(LazyAccessor):
    def __init__(
        self,
        module,
        node_name: Optional[str] = None,
        node_namespace: Optional[str] = None,
    ):
        LazyAccessor.__init__(self)
        self._module = module
        self.node_name = node_name
        self.node_namespace = node_namespace

    def get_node_def(self):
        """
        Returns an instance of the first RosNode subclass within the module.
        """
        for _, obj in self._module.__dict__.items():
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
) -> RosImport:
    """
    Defines a node dependency.

    Args:
        module (ModuleType): A module that contains a node definition for the dependency.
        node_name (Optional[str]): Does not need to be provided if the nodes are launched from a launch file and linked properly.
        node_namespace (Optional[str]): Does not need to be provided if the nodes are launched from a launch file and linked properly.
    """
    return RosImport(module, node_name, node_namespace)
