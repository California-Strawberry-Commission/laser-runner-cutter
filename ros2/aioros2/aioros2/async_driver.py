import asyncio
import dataclasses
import rclpy

from .decorators.deferrable_accessor import DeferrableAccessor
from .decorators import RosDefinition
from .decorators.service import RosService
from .decorators.topic import RosTopic
from .decorators.subscribe import RosSubscription
from .decorators.import_node import RosImport
from .decorators.action import RosAction
from .decorators.timer import RosTimer
from .decorators.params import RosParams


class AsyncDriver:
    """Base class for all adapters"""
    
    def __init__(self, async_node, logger):
        self.log = logger
        self._n = async_node
        
        # self._n.params = self._attach_params_dataclass(self._n.params)
        self._loop = asyncio.get_running_loop()

    # def __getattr__(self, attr):
    #     """Lookup unknown attrs on node definition. Effectively extends node def"""
    #     try:
    #         return getattr(self._n, attr)
    #     except AttributeError:
    #         raise AttributeError(f"Attribute >{attr}< was not found in either [{type(self).__qualname__}] or [{type(self._n).__name__}]")


    def _get_ros_definitions(self):
        from .decorators.import_node import RosImport

        node = self._n

        a = [
            (attr, getattr(node, attr))
            for attr in dir(node)
            if isinstance(getattr(node, attr), RosDefinition)
        ]

        # Make sure RosImports are processed first
        a.sort(key=lambda b: type(b[1]) != RosImport)

        # Process topics last to prevent function version from shadowing definition
        a.sort(key=lambda b: type(b[1]) == RosTopic)

        return a
    
    def _attach(self):
        # Attachers create an implementation for the passed handler which is assigned
        # to that handler's name.
        attachers = {
            # ros_type_e.ACTION: self._attach_action,
            RosService: self._attach_service,
            RosSubscription: self._attach_subscriber,
            RosTopic: self._attach_publisher,
            RosImport: self._process_import,
            RosAction: self._attach_action,
            RosTimer: self._attach_timer,
            RosParams: self._attach_params
        }

        for attr, definition in self._get_ros_definitions():
            setattr(self, attr, attachers[type(definition)](attr, definition))
            
    def _warn_unimplemented(self, readable_name, fn_name):
        self.log.warn(f"Failed to initialize >{readable_name}< because >{fn_name}< is not implemented in driver >{self.__class__.__qualname__}<")

    def _process_import(self, attr, d):
        self._warn_unimplemented("import", "_process_import")

    def _attach_service(self, attr, d):
        self._warn_unimplemented("service", "_attach_service")

    def _attach_subscriber(self, attr, d):
        self._warn_unimplemented("subscriber", "_attach_subscriber")
    
    def _attach_publisher(self, attr, d):
        self._warn_unimplemented("topic publisher", "_attach_publisher")

    def _attach_action(self, attr, d):
        self._warn_unimplemented("action", "_attach_action")
    
    def _attach_param_handler(self, attr, d):
        self._warn_unimplemented("param handler", "_attach_param_handler")
    
    def _attach_timer(self, attr, d):
        self._warn_unimplemented("timer", "_attach_timer")
        
    def _attach_params(self, attr, d):
        self._warn_unimplemented("params", "_attach_params")

dataclass_ros_map = {
    bool: 1,
    int: 2,
    float: 3,
    str: 4,
    bytes: 5,
    "list[bool]": 6,
    "list[int]": 7,
    "list[float]": 8,
    "list[str]": 9,
}

dataclass_ros_enum_map = {
    bool: rclpy.Parameter.Type.BOOL,
    int: rclpy.Parameter.Type.INTEGER,
    float: rclpy.Parameter.Type.DOUBLE,
    str: rclpy.Parameter.Type.STRING,
    bytes: rclpy.Parameter.Type.BYTE_ARRAY,
    "list[bool]": rclpy.Parameter.Type.BOOL_ARRAY,
    "list[int]": rclpy.Parameter.Type.INTEGER_ARRAY,
    "list[float]": rclpy.Parameter.Type.DOUBLE_ARRAY,
    "list[str]": rclpy.Parameter.Type.STRING_ARRAY,
}

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

  