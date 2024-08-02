from collections import namedtuple
from typing import Any, Union
from enum import Enum

Returnable = namedtuple("Returnable", ["args", "kwargs", "tag"])

def result(*args, **kwargs):
    return Returnable(args, kwargs, tag="result")

def feedback(*args, **kwargs):
    return Returnable(args, kwargs, tag="feedback")

class PreMarshalError(ValueError):
    pass

# Converts data in compatible formats into the passed IDL.
# Accepts raw IDLs, `Returnable` namedTuples, and dictionaries.
def marshal_returnable_to_idl(returnable: Union[Returnable, tuple, Any], idl, enforce_tag=None):
    def _marshal(args, kwargs):
        try:
            return idl(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not marshal returnable >{returnable}< into >{idl}<")
        
    # Handle returnable helpers
    if isinstance(returnable, Returnable):
        if (enforce_tag is not None) and returnable.tag != enforce_tag:
            raise PreMarshalError(f"Returnable type does not match the enforced returnable type >{enforce_tag}<")
        
        return _marshal(returnable.args, returnable.kwargs)

    # Handle raw, user-constructed IDL returns
    elif hasattr(returnable, "_fields_and_field_types"):
        if not isinstance(returnable, idl):
            raise PreMarshalError(f"Passed IDL object >{returnable}< does not match the required IDL >{idl}<!")

        return returnable
    
    # Handle dict inputs helpers
    elif isinstance(returnable, dict):
        if enforce_tag is not None:
            raise PreMarshalError(f"Cannot marshal from a dictionary when a tag (>{enforce_tag}<) is enforced. Pass either an IDL instance or the appropriate returnable.")
        
        return _marshal([], returnable)

    else:
        raise ValueError(f"Non-returnable, non-idl, non-dict value >{returnable}< returned/yeilded! This function must return a dict, and IDL, or a `Returnable`!")

