import inspect
from ._decorators import decorate_handler, ros_type_e

# https://stackoverflow.com/questions/11731136/class-method-decorator-with-self-arguments
# TODO: Check types, check annotated return value if exists.
def _check_service_handler_signature(fn, srv):
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
        _check_service_handler_signature(fn, srv_idl)
        decorate_handler(fn, ros_type_e.SERVICE, ros_idl=srv_idl, ros_namespace=namespace)
        return fn

    return _service

