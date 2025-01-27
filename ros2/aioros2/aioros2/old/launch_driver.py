from __future__ import annotations

from typing import Optional

from launch_ros.actions import Node as RosLaunchNode
from rclpy.logging import get_logger

from aioros2.async_driver import AsyncDriver
from aioros2.decorators.node import RosNode

# TODO: Validate all imports are linked before allowing execution ??


class ImportLinker:
    def __init__(self, attr, cb) -> None:
        self._cb = cb
        self._attr = attr

    def link(self, node: LaunchNode):
        self._cb(self._attr, node)


class LaunchNode(RosLaunchNode, AsyncDriver):
    def __init_rclpy_node(self):
        # Override parameters kwargs with launch params in addition to passed params
        self._kwargs["parameters"] = self._parameters + (
            self._kwargs["parameters"] if ("parameters" in self._kwargs) else []
        )

        RosLaunchNode.__init__(
            self,
            package=self._package,
            executable=self._executable,
            name=self._name,
            namespace=self._namespace,
            **self._kwargs,
        )

    def __init__(
        self,
        node_package,
        name: Optional[str] = None,
        namespace: Optional[str] = "/",
        **kwargs,
    ):
        # TODO: Check this logic to make sure it actually extracts the top level package correctly in all cases
        self._package = node_package.__name__.split(".")[0]

        self._namespace = namespace
        self._name = name
        self._parameters = []
        self._kwargs = kwargs

        # Find the RosNode definition within the passed module and create an instance
        for _, obj in node_package.__dict__.items():
            if (
                isinstance(obj, type)
                and obj is not RosNode
                and issubclass(obj, RosNode)
            ):
                node_def = obj()

        if not node_def:
            raise ImportError(
                f"Launched module >{node_package.__name__}< does not contain a valid aioros2 node. Make sure your aioros2 class is annotated with >@node<!"
            )

        # Set defaults for node name and namespace if not provided
        if self._name is None:
            self._name = (
                node_def._aioros2_name
                if node_def._aioros2_name
                else node_def._aioros2_executable
            )
            self.log_warn(
                f"Name not provided for node of package >{node_package.__name__}<. Setting default name >{self._name}<!"
            )
        if self._namespace is None:
            self._namespace = (
                node_def._aioros2_namespace if node_def._aioros2_namespace else "/"
            )

        # Append parameter overrides from node_def
        if node_def._aioros2_parameter_overrides:
            self._parameters += node_def._aioros2_parameter_overrides

        self._executable = node_def._aioros2_executable
        self.__init_rclpy_node()
        AsyncDriver.__init__(
            self, node_def, get_logger(f"LAUNCH-{namespace}-{name}"), name, namespace
        )

        self.log_debug(f"Launching node >{self._package}< >{self._executable}<")

        self._attach()

    def _link_node(self, attr, other_node: "LaunchNode"):
        # TODO: extract naming logic somewhere else. This is duplicated
        # in server_driver._process_imports
        node_name_param_name = f"{attr}.name"
        node_namespace_param_name = f"{attr}.ns"

        self._parameters.append({node_name_param_name: other_node._name})
        self._parameters.append({node_namespace_param_name: other_node._namespace})

        self.__init_rclpy_node()

    def _process_import(self, attr, imp):
        return ImportLinker(attr, self._link_node)

    # Dummy attachment implementations to silence warnings
    def _attach_service(self, attr, ros_service):
        pass

    def _attach_subscriber(self, attr, ros_sub):
        pass

    def _attach_publisher(self, attr, ros_topic):
        pass

    def _attach_action(self, attr, ros_action):
        pass

    def _attach_timer(self, attr, ros_timer):
        pass

    def _attach_params(self, attr, ros_params):
        pass

    def _attach_param_subscription(self, attr, ros_param_sub):
        pass

    def _process_start(self, attr, ros_start):
        pass
