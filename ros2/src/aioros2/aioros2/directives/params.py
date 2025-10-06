import asyncio
import dataclasses
from typing import Any, List, Optional, TypeVar

from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from varname import varname

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.util import catch


class RosParams(RosDirective):
    _node: Optional[Node]
    _loop: Optional[asyncio.BaseEventLoop]
    # Backing instance of the dataclass that we'll read/write to.
    _dataclass_instance: Any
    # Namespace of the params. For each field in the dataclass, the corresponding ROS param will be
    # declared as namespace.name
    _namespace: str

    def __init__(self, params_dataclass: Any, namespace: str):
        # We set instance variables like this since we're overriding __setattr__ and don't want to
        # use it here.
        # Based on https://stackoverflow.com/a/58676807/16238567
        vars(self).update(
            dict(
                _node=None,
                _loop=None,
                _dataclass_instance=params_dataclass(),
                _namespace=namespace,
            )
        )

    def __setattr__(self, name: str, value: Any):
        setattr(self._dataclass_instance, name, value)
        # Sync to node
        param_path = self._namespace + "." + name
        try:
            parameter = self._node.get_parameter(param_path)
            new_parameter = Parameter(parameter.name, parameter.type_, value)
            self._loop.run_in_executor(None, self._node.set_parameters, [new_parameter])
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._dataclass_instance, name)

    def server_impl(
        self,
        node: Node,
        nodeinfo: NodeInfo,
        loop: asyncio.BaseEventLoop,
    ):
        # We set instance vars like this since we're overriding __setattr__ and don't want to
        # use it here.
        self.__dict__["_node"] = node
        self.__dict__["_loop"] = loop

        fields = [
            (
                f.name,
                (
                    f.default_factory()
                    if f.default_factory is not dataclasses.MISSING
                    else f.default
                ),
                f.type,
            )
            for f in dataclasses.fields(self._dataclass_instance)
        ]

        for name, default_val, _ in fields:
            param_path = self._namespace + "." + name

            # Declare and get current param value
            node.declare_parameter(param_path, default_val)
            val = node.get_parameter(param_path).value

            # Update internal dataclass with current val
            setattr(self._dataclass_instance, name, val)

        @catch(node.get_logger().log)
        def callback(params: List[Parameter]) -> SetParametersResult:
            # This callback gets called on any param change triggered on this node.
            # TODO: sync the updates to _dataclass_instance and call param change listeners
            return SetParametersResult(successful=True)

        # TODO: sync ROS 2 param updates to dataclass instance
        # node.add_on_set_parameters_callback(callback)

    def client_impl(
        self,
        node: Node,
        nodeinfo: NodeInfo,
        loop: asyncio.BaseEventLoop,
    ):
        # Not used on clients
        return


T = TypeVar("T")


def params(params_dataclass: T) -> T:
    """
    Defines a parameters object that can be published to by the node.

    Args:
        params_dataclass: Class decorated with `@dataclass` that defines the params object.
    """
    return RosParams(params_dataclass, varname())
