from collections.abc import Iterable
from typing import Any

# Global which should be set after initial load is completed, and main() is running
# Accesses performed after this is set to True will create and access deferred objects
deferrables_frozen=False

class _DeferrableAccessor:
    """Magic class that can cache access to an object without evaluating it.
    Can be used to define access paths before they actually exist.
    """

    
    def __init__(self, create_self, initial_path) -> None:
        self.__instance = None
        self.__create_self = create_self
        self.__path = initial_path

    def __get_instance(self):
        if not self.__instance:
            self.__instance = self.__create_self()
        
        print(self.__dict__)

        return self.__instance
    
    def __getattr__(self, attr: str) -> Any:
        if not deferrables_frozen:
            return DeferrableAccessor(self.__path + [attr])
        else:
            return getattr(self.__resolve(), attr)
    
    def __setattr__(self, attr: str, val) -> Any:
        
        if "__instance" in attr or not deferrables_frozen:
            self.__dict__[attr] = val
        else:
            setattr(self.__resolve(), attr, val)
        
    def __repr__(self) -> str:
        return f"{self.__get_instance()} - path: " + ".".join(self.__path)
    
    def __resolve(self):
        try:
            v = self.__get_instance()
            for seg in self.__path:
                v = getattr(v, seg)

        except AttributeError:
            raise AttributeError(f"Could not resolve deferred access on >{self}<")
    
        return v

# Base class allows any class to implement deferred access to itself
class DeferrableAccessor:
    def __init__(self, create_self) -> None:
        self.__instance = None
        self.__create_self = create_self

    def __getattr__(self, attr: str) -> Any:
        if not deferrables_frozen:
            return _DeferrableAccessor(self.__create_self, [attr])
        else:
            return getattr(self.__get_instance(), attr)
        
    def __setattr__(self, attr: str, val) -> Any:
        if "__instance" in attr or not deferrables_frozen:
            self.__dict__[attr] = val
        else:
            setattr(self.__get_instance(), attr, val)

        
    def __get_instance(self):
        if not self.__instance:
            self.__instance = self.__create_self()

        return self.__instance
    
    def __dir__(self):
        return dir(self.__get_instance())       