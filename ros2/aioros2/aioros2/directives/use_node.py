import asyncio
from types import ModuleType
from typing import Any, TypeVar

import rclpy.node
from rclpy.exceptions import ParameterUninitializedException
from rclpy.parameter import Parameter
from varname import varname

from aioros2.directives.directive import NodeInfo, RosDirective
from aioros2.util import duplicate_module, get_module_ros_directives


class RosUseNode(RosDirective):
    """
    aioros2's import. Allows node linkage at launch time and multi-instance imports.

    ```
    import testnodes.talker
    talker_node1 = use(testnodes.talker)
    talker_node2 = use(testnodes.talker)

    @subscribe(talker_node1.topic)
    def handle_topic1():
        ...

    @subscribe(talker_node2.topic)
    def handle_topic2():
        ...
    ```
    """

    # A unique instance of the module this `use` directive wraps
    # All variables, functions, etc within this module are different instances
    # to those accessible through an `import` statement.
    _module_instance: Any

    # The default module, received through an `import` statement.
    _module: ModuleType

    # Name of the variable this `use` is assigned to.
    _param_name: str

    def __init__(
        self,
        module: ModuleType,
        param_name: str,
    ):
        # We set instance variables like this since we're overriding __setattr__ and don't want to
        # use it here.
        # Based on https://stackoverflow.com/a/58676807/16238567
        vars(self).update(
            dict(_param_name=param_name, _module=module, _module_instance=module)
        )

    def __getattr__(self, name: str):
        try:
            return getattr(self._module_instance, name)
        except AttributeError:
            # For circular imports, modules may only be partially initialized. To ensure references
            # that are evaluated at module load time (for example, topic references in the subscribe
            # decorator) work, we just return None here, and ensure that we use `Deferrable` and
            # resolve the reference when we process the directive.
            return None

    def __setattr__(self, name: str, value: Any):
        setattr(self._module_instance, name, value)

    def server_impl(
        self, node: rclpy.node.Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop
    ):
        """
        When loaded by a server node, all use directives have their modules reloaded to create
        unique, independent instances.
        """
        try:
            # Load name and namespace of referenced node from parameters
            name_param = self._param_name + ".name"
            namespace_param = self._param_name + ".namespace"

            # The name param is required to be defined ahead of time at launch
            node.declare_parameter(name_param, Parameter.Type.STRING)
            # The namespace param is optional, and will default to "/"
            node.declare_parameter(namespace_param, "/")

            name = node.get_parameter(name_param).value
            namespace = node.get_parameter(namespace_param).value

            # Reinstantiate the module to get independant funcs and vars.
            clone = duplicate_module(self._module)
            # We set _module_instance like this since we're overriding __setattr__ and don't
            # want to use it here.
            self.__dict__["_module_instance"] = clone

            # Initialize the instance as a client.
            directives = get_module_ros_directives(clone)

            for directive in directives:
                directive.client_impl(node, NodeInfo(namespace, name), loop)

        except ParameterUninitializedException:
            node.get_logger().error(
                f"Could not link {self._param_name}"
                f" because `{self._param_name}.namespace` "
                f"or `{self._param_name}.name` params were "
                f"uninitialized"
            )

    def client_impl(
        self, node: rclpy.node.Node, nodeinfo: NodeInfo, loop: asyncio.BaseEventLoop
    ):
        # More than 1 level of removal is not supported. In other words, node clients will not
        # instantiate clients for their `use`d nodes.
        return


U = TypeVar("U", bound=ModuleType)


def use(module: U) -> U:
    return RosUseNode(module, varname())
