from typing import Callable, TypeVar


T = TypeVar("T")
def import_node(l: Callable[[], T]) -> T:
    pass