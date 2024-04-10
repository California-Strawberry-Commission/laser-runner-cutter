from typing import Any


class Self:
    """Magic class that caches any accesses without evaluating them.
    Can be used to define access paths before they actually exist.
    
    Also provides helpers to process the access on another class
    """
    
    def __init__(self, path=[]) -> None:
        self.__path = path
    
    def __getattr__(self, attr: str) -> Any:
        return Self(self.__path + [attr])
    
    def __repr__(self) -> str:
        return "Self Object - path: " + ".".join(self.__path)
    
    def resolve(self, otherSelf):
        v = otherSelf
        for seg in self.__path:
            v = getattr(v, seg)
            
        return v
    
self = Self()