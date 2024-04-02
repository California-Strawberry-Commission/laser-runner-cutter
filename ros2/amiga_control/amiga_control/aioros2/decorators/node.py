from typing import TypeVar, Callable
from ..client_driver import ClientDriver
from ..server_driver import ServerDriver
from ..util import to_snake


PARAMS_T = TypeVar("PARAMS_T")
CLASS_T = TypeVar("CLASS_T")


def node(params_class: PARAMS_T = None) -> Callable[[CLASS_T], CLASS_T]:
    def _rosnode(cls: CLASS_T):

        class _RosNode(cls):
            def __init__(self, name=to_snake(cls.__name__), params=params_class() if params_class is not None else None):
                self.node_name = name
                self.params = params
                cls.__init__(self)

        _RosNode.__name__ = cls.__name__
        return _RosNode

    return _rosnode
