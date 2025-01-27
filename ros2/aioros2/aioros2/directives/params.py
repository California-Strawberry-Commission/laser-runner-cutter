import asyncio
import dataclasses
from typing import TypeVar

from rclpy.node import Node
from rclpy.parameter import Parameter, parameter_value_to_python
from varname import varname

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.mappings import dataclass_ros_enum_map


class RosParams(RosDirective):
    # Include these here for typing
    _dclass = None
    _base = None

    def __init__(self, params_dclass, base_name) -> None:
        # Because we're defining a custom __setattr__ and don't want to use it here, bypass setattr
        # Based on https://stackoverflow.com/a/58676807/16238567
        vars(self).update(
            dict(
                _dclass=params_dclass(),
                _base=base_name,
            )
        )

    def __setattr__(self, name: str, value) -> None:
        return setattr(self._dclass, name, value)

    # Returns array of listeners that caller can add to.
    def __getattr__(self, attr):
        return getattr(self._dclass, attr)

    def server_impl(
        self,
        node: Node,
        nodeinfo: NodeInfo,
        loop: asyncio.BaseEventLoop,
    ):
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
            for f in dataclasses.fields(self._dclass)
        ]

        for name, default_val, _ in fields:
            param_path = self._base + "." + name

            # Declare and get current param value
            node.declare_parameter(param_path, default_val)
            val = node.get_parameter(param_path).value

            # Update internal dataclass with current val
            setattr(self._dclass, name, val)

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
