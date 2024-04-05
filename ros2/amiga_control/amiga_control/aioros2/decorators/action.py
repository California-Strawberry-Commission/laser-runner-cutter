import inspect
from ._decorators import decorate_handler, ros_type_e

def _check_action_handler_signature(fn, act):
    # ['Feedback', 'Goal', 'Impl', 'Result']
    fn_inspection = inspect.signature(fn)
    fn_dict = fn_inspection.parameters
    fn_params = set(fn_dict)
    
    idl_dict = act.Goal.get_fields_and_field_types()
    idl_params = set(idl_dict.keys())

    fn_params.discard("self")

    if fn_params != idl_params:
        raise RuntimeError(
            "Service handler parameters do not match those in the IDL format! "
            "Make sure that the function parameter names match those in the IDL!\n"
            f"Handler: {fn.__name__} -> \t{fn_params if len(fn_params) else 'NO ARGUMENTS'}\n"
            f"    IDL: {act.__name__} -> \t{idl_params}"
        )

def action(namespace, act_idl):
    def _action(fn):
        _check_action_handler_signature(fn, act_idl)
        
        return decorate_handler(fn, ros_type_e.ACTION, idl=act_idl, namespace=namespace)


    return _action
