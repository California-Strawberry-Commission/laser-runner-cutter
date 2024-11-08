import inspect
from ._decorators import RosDefinition, idl_to_kwargs

# https://stackoverflow.com/questions/11731136/class-method-decorator-with-self-arguments


class RosService(RosDefinition):
    def __init__(self, path, idl, handler):
        if not hasattr(idl, "Request"):
            raise TypeError(
                "Passed object is not a service-compatible IDL object! Make sure it isn't a topic or action IDL."
            )

        self._check_service_handler_signature(handler, idl)
        self.path = path
        self.idl = idl
        self.handler = handler

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


def service(namespace, srv_idl):
    def _service(fn):
        return RosService(namespace, srv_idl, fn)

    return _service
