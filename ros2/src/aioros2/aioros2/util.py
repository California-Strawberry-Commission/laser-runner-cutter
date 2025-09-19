import importlib
import inspect
import re
import traceback
from types import ModuleType

from rclpy.logging import LoggingSeverity

from aioros2.directives.directive import RosDirective


# Decorate sync callbacks to catch errors into the specified print function.
def catch(log_fn, return_val=None):
    def _catch(fn):
        def _safe_exec(*args, **kwargs):
            try:
                return fn(*args, **kwargs)

            except Exception:
                log_fn(traceback.format_exc(), LoggingSeverity.ERROR)
                return return_val

        return _safe_exec

    return _catch


def snake_to_camel_case(snake_str: str) -> str:
    # https://stackoverflow.com/questions/19053707/converting-snake-case-to-lower-camel-case-lowercamelcase
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def camel_to_snake_case(camel_str: str) -> str:
    camel_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()


def get_caller_module(skip=0):
    stack = inspect.stack()

    start_idx = (
        2 + skip
    )  # Always omit this function and the calling function from the stack.
    caller_frame = stack[start_idx][0]

    caller_module = inspect.getmodule(caller_frame)

    return caller_module


def get_module_ros_directives(d) -> list[RosDirective]:
    """
    Returns all found directives within the passed module
    """
    if isinstance(d, ModuleType):
        d = d.__dict__

    if not isinstance(d, dict):
        return []

    return [
        d[k] for k in d if not k.startswith("__") and isinstance(d[k], RosDirective)
    ]


# https://stackoverflow.com/a/57249901/16238567
def duplicate_module(module):
    """
    Creates a deep copy of the passed python module by loading it again.
    """
    fullname = module.__name__
    spec = importlib.util.find_spec(fullname)
    clone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clone)

    return clone


def marshal_to_idl(idl, *args, **kwargs):
    """
    Converts kwargs into the passed IDL.

    Special behavior for non-kwargs args:
    If passed an instantiated IDL, it will be returned as-is.
    If passed a dictionary as an argument, it will be used instead of kwargs.
    """
    # If passed an already instantiated IDL object, return it
    if len(args) == 1 and isinstance(args[0], idl):
        return args[0]

    # Handle dictionary arg.
    elif len(args) == 1 and isinstance(args[0], dict):
        return idl(**args[0])

    return idl(**kwargs)


def idl_to_kwargs(req):
    """
    Converts any instantiated rclpy IDL object into a dict,
    passable as kwargs
    """
    msg_keys = req.get_fields_and_field_types().keys()
    return {k: getattr(req, k) for k in msg_keys}
