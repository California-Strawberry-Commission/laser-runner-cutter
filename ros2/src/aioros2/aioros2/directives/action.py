import asyncio
import inspect

from rclpy.node import Node

from aioros2.directives.directive import RosDirective
from aioros2.exception import AioRos2Exception


class RosAction(RosDirective):
    def __init__(self, path, idl, fn) -> None:
        self._check_action_handler_signature(fn, idl)

        self._path = path
        self._idl = idl
        self._fn = fn

        self._client_mode = False

    async def __call__(self, *args: any, **kwargs: any) -> any:
        if self._client_mode:
            raise AioRos2Exception("Cannot call another node's action function.")

        return await self._fn(*args, **kwargs)

    def server_impl(self, node: Node, nodeinfo, loop: asyncio.BaseEventLoop):
        # TODO
        pass

    def client_impl(self, node: Node, nodeinfo, loop: asyncio.BaseEventLoop):
        # TODO
        self._client_mode = True
        pass

    def _check_action_handler_signature(self, fn, act):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("Action handlers must be async.")

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
