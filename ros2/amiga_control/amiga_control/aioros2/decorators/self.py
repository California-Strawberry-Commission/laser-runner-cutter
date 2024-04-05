from typing import Any


class Self:
    """Magic class that caches any accesses without evaluating them.
    Can be used to define access paths before they actually exist.
    
    Also provides helpers to process the access on another class
    """
    
    def __init__(self, path=[]) -> None:
        self._path = path
    
    def __getattr__(self, name: str) -> Any:
        return Self(self._path + [name])
    
    def __repr__(self) -> str:
        return "Self Object - path: " + ".".join(self._path)
    
self = Self()