import asyncio
import inspect
from typing import Any, Optional

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.expand_topic_name import expand_topic_name
from rclpy.node import Node

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.returnable import marshal_returnable_to_idl
from aioros2.util import catch, idl_to_kwargs


class RosService(RosDirective):
    def __init__(self, name: str, idl: Any, fn):
        if not hasattr(idl, "Request"):
            raise TypeError(
                "Passed object is not a service-compatible IDL object! Make sure it isn't a topic or action IDL."
            )

        self._check_service_handler_signature(fn, idl)

        self._path = name
        self._idl = idl
        self._fn = fn

        self._client_mode = False
        self._loop: Optional[asyncio.BaseEventLoop] = None
        self._node: Optional[Node] = None
        self._nodeInfo = NodeInfo(None, None)
        self._client: Client = None

    async def __call__(self, *args: any, **kwargs: any) -> any:
        if self._client_mode:
            # When another node's service function is called, we need to create a client on the node
            # and call it.
            client = self._get_client()
            request = self._idl.Request(*args, **kwargs)
            return await client.call_async(request)
        else:
            return await self._fn(*args, **kwargs)

    def server_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        self._node = node
        self._nodeInfo = nodeinfo
        self._loop = loop

        @catch(node.get_logger().log, self._idl.Response())
        def callback(req, result):
            kwargs = idl_to_kwargs(req)

            # Call handler function. This callback is called from another thread, so we need to
            # use run_coroutine_threadsafe
            user_return = asyncio.run_coroutine_threadsafe(
                self._fn(node, **kwargs), loop
            ).result()

            return marshal_returnable_to_idl(user_return, self._idl.Response)

        node.create_service(self._idl, self._path, callback)

    def client_impl(self, node: Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop):
        self._client_mode = True
        self._node = node
        self._nodeInfo = nodeinfo
        self._loop = loop
        return

    def _check_service_handler_signature(self, fn, srv):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("Service handlers must be async.")

        fn_name = fn.__name__
        fn_inspection = inspect.signature(fn)
        fn_dict = fn_inspection.parameters
        # The first param is the node, so remove it for the purposes for checking params against
        # the IDL
        fn_params = set(fn_dict)
        fn_params.discard("node")

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

    def _get_fully_qualified_path(self) -> str:
        return expand_topic_name(
            self._path, self._nodeInfo.name, self._nodeInfo.namespace
        )

    def _get_client(self) -> Client:
        if not self._client:
            self._client = self._node.create_client(
                self._idl,
                self._get_fully_qualified_path(),
                callback_group=ReentrantCallbackGroup(),
            )

        return self._client


# Decorator
def service(name: str, idl: Any):
    """
    A function decorator for a function that will be run on a service call.

    Args:
        name (str): Service name. Relative and private names are accepted and will be resolved appropriately.
        idl (Any): ROS 2 message type associated with the service.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _service(fn) -> RosService:
        return RosService(name, idl, fn)

    return _service
