from typing import Generic, Type, TypeVar, Callable, Cl
from ..client_driver import ClientDriver
from ..server_driver import ServerDriver
from ..util import to_snake


PARAMS_T = TypeVar("PARAMS_T")
CLASS_T = TypeVar("CLASS_T")

class RosNode():
    def client(self) -> ClientDriver:
        return ClientDriver(self)
    
    def server(self) -> ServerDriver:
        return ServerDriver(self)
    
def node(params_class: PARAMS_T) -> Callable[[CLASS_T], CLASS_T]:
    def _rosnode(cls: CLASS_T):
        extendedClass = type("RosNode", (RosNode, cls), {})

        extendedClass.__init__.__defaults__ = params_class(), to_snake(cls.__name__)
    
        return extendedClass

    return _rosnode

