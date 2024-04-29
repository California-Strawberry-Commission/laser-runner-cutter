import inspect
from ._decorators import RosDefinition

class RosAction(RosDefinition):
    def __init__(self, path, idl, handler) -> None:
        self._check_action_handler_signature(handler, idl)
        
        self.path = path
        self.idl = idl
        self.handler = handler

    def _check_action_handler_signature(self, fn, act):
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
        return RosAction(namespace, act_idl, fn)

    return _action
