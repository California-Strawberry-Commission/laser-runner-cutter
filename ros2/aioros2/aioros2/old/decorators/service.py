import inspect
from typing import Any


class RosService:
    def __init__(self, name, idl, func):
        if not hasattr(idl, "Request"):
            raise TypeError(
                "Passed object is not a service-compatible IDL object! Make sure it isn't a topic or action IDL."
            )

        self._check_service_handler_signature(func, idl)
        self.path = name
        self.idl = idl
        self.handler = func

    def _check_service_handler_signature(self, fn, srv):
        fn_name = fn.__name__
        fn_inspection = inspect.signature(fn)
        fn_dict = fn_inspection.parameters
        fn_params = set(fn_dict)
        fn_params.discard("self")

        idl_dict = srv.Request.get_fields_and_field_types()
        idl_params = set(idl_dict.keys())

        if fn_params != idl_params:
            raise RuntimeError(
                f"PROBLEM WITH SERVICE >{fn_name}<\n"
                f"Service handler parameters do not match those in the IDL format!\n"
                f"Make sure that the function parameter names match those in the IDL!\n"
                f"Handler: {fn_name} -> \t{fn_params if len(fn_params) else 'NO ARGUMENTS'}\n"
                f"    IDL: {fn_name} -> \t{idl_params}"
            )


def service(name: str, idl: Any):
    """
    A function decorator for a function that will be run on a service call.

    Args:
        name (str): Service name. Relative and private names are accepted and will be resolved appropriately.
        idl (Any): ROS 2 message type associated with the service.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _service(func):
        if not inspect.isfunction(func):
            raise TypeError("This decorator can only be applied to functions.")

        return RosService(name, idl, func)

    return _service
