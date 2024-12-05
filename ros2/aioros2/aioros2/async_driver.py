import asyncio
import logging
import types
from collections import OrderedDict
from inspect import getmembers
from typing import Any, List, Optional

import rclpy

from aioros2.decorators.action import RosAction
from aioros2.decorators.import_node import RosImport
from aioros2.decorators.param_subscription import RosParamSubscription
from aioros2.decorators.params import RosParams
from aioros2.decorators.service import RosService
from aioros2.decorators.start import RosStart
from aioros2.decorators.subscribe import RosSubscription
from aioros2.decorators.timer import RosTimer
from aioros2.decorators.topic import RosTopic


class AsyncDriver:
    """Base class for all adapters"""

    def __getattr__(self, attr):
        if not hasattr(self._node_def, attr):
            raise AttributeError(
                f"Attr >{attr}< not found in either driver or definition class"
            )

        value = getattr(self._node_def, attr)

        # Rebind self-bound definition functions to this driver
        if isinstance(value, types.MethodType):
            value = types.MethodType(value.__func__, self)

        # Cache result for future accesses to bypass this
        # getattr
        setattr(self, attr, value)

        return value

    def __init__(
        self,
        node_def: Any,
        logger: logging.Logger,
        node_name: str,
        node_namespace: Optional[str],
    ):
        self._logger = logger
        self._node_def = node_def
        self._node_name = node_name
        self._node_namespace = node_namespace if node_namespace is not None else "/"

        # self._node_def.params = self._attach_params_dataclass(self._node_def.params)
        self._loop = asyncio.get_running_loop()

    @property
    def node_def(self):
        return self._node_def

    @property
    def node_name(self):
        return self._node_name

    @property
    def node_namespace(self):
        return self._node_namespace

    async def run_executor(self, fn, *args, **kwargs):
        """Runs a synchronous function in an executor"""
        return await self._loop.run_in_executor(None, fn, *args, **kwargs)

    def run_coroutine(self, fn, *args, **kwargs):
        """Runs asyncio code from ANOTHER SYNC THREAD"""

        # https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.call_soon_threadsafe
        return asyncio.run_coroutine_threadsafe(fn(*args, **kwargs), self._loop)

    def log_debug(self, msg: str):
        self._logger.debug(msg)

    def log_warn(self, msg: str):
        self._logger.warning(msg)

    def log_error(self, msg: str):
        self._logger.error(msg)

    def log(self, msg: str):
        self._logger.info(msg)

    def _attach(self):
        # Attachers create an implementation for the passed handler which is assigned
        # to that handler's name.
        att = OrderedDict()

        # Defines load order.
        att[RosImport] = self._process_import  # Process imports first
        att[RosTopic] = (
            self._attach_publisher
        )  # Attach topics second b/c they are referenced by services/subscribers/etc.
        att[RosService] = self._attach_service
        att[RosSubscription] = self._attach_subscriber
        att[RosAction] = self._attach_action
        att[RosTimer] = self._attach_timer
        att[RosParams] = self._attach_params
        att[RosParamSubscription] = self._attach_param_subscription
        att[RosStart] = (
            self._process_start
        )  # Process starts last b/c they directly go into asyncio loop

        for ros_element, attacher in att.items():
            for attr, definition in getmembers(
                self._node_def, lambda v: isinstance(v, ros_element)
            ):
                setattr(self, attr, attacher(attr, definition))

    def _warn_unimplemented(self, readable_name: str, fn_name: str):
        self._logger.warning(
            f"Failed to initialize >{readable_name}< because >{fn_name}< is not implemented in driver >{self.__class__.__qualname__}<"
        )

    def _process_import(self, attr: str, ros_import: RosImport):
        self._warn_unimplemented("import", "_process_import")

    def _attach_service(self, attr: str, ros_service: RosService):
        self._warn_unimplemented("service", "_attach_service")

    def _attach_subscriber(self, attr: str, ros_sub: RosSubscription):
        self._warn_unimplemented("subscriber", "_attach_subscriber")

    def _attach_publisher(self, attr: str, ros_topic: RosTopic):
        self._warn_unimplemented("topic publisher", "_attach_publisher")

    def _attach_action(self, attr: str, ros_action: RosAction):
        self._warn_unimplemented("action", "_attach_action")

    def _attach_timer(self, attr: str, ros_timer: RosTimer):
        self._warn_unimplemented("timer", "_attach_timer")

    def _attach_params(self, attr: str, ros_params: RosParams):
        self._warn_unimplemented("params", "_attach_params")

    def _attach_param_subscription(
        self, attr: str, ros_param_sub: RosParamSubscription
    ):
        self._warn_unimplemented("param subscription", "_attach_param_subscription")

    def _process_start(self, attr: str, ros_start: RosStart):
        self._warn_unimplemented("start", "_process_start")


# Maps python types to a ROS parameter integer enum
dataclass_ros_map = {
    bool: 1,
    int: 2,
    float: 3,
    str: 4,
    bytes: 5,
    List[bool]: 6,
    List[int]: 7,
    List[float]: 8,
    List[str]: 9,
}

# Maps python types to a ROS parameter enum
dataclass_ros_enum_map = {
    bool: rclpy.Parameter.Type.BOOL,
    int: rclpy.Parameter.Type.INTEGER,
    float: rclpy.Parameter.Type.DOUBLE,
    str: rclpy.Parameter.Type.STRING,
    bytes: rclpy.Parameter.Type.BYTE_ARRAY,
    List[bool]: rclpy.Parameter.Type.BOOL_ARRAY,
    List[int]: rclpy.Parameter.Type.INTEGER_ARRAY,
    List[float]: rclpy.Parameter.Type.DOUBLE_ARRAY,
    List[str]: rclpy.Parameter.Type.STRING_ARRAY,
}

# Maps ROS types to a corresponding attribute containing the
# typed value
ros_type_getter_map = {
    1: "bool_value",
    2: "integer_value",
    3: "double_value",
    4: "string_value",
    5: "byte_array_value",
    6: "bool_array_value",
    7: "integer_array_value",
    8: "double_array_value",
    9: "string_array_value",
}
