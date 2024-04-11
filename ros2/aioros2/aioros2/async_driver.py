import asyncio
import dataclasses
import rclpy

from .decorators.self import Self
from .decorators import RosDefinition
from .decorators.service import RosService
from .decorators.topic import RosTopic
from .decorators.subscribe import RosSubscription
from .decorators.import_node import RosImport
from .decorators.action import RosAction
from .decorators.timer import RosTimer
from .decorators.param import RosParamHandler

getter_map = {
    str: "string_value",
    int: "integer_value",
    bool: "bool_value",
    float: "double_value",
    "list[int]": "byte_array_value",
    "list[bool]": "bool_array_value",
    "list[int]": "integer_array_value",
    "list[float]": "double_array_value",
    "list[str]": "string_array_value",
}

class AsyncDriver(rclpy.node.Node):
    """Base class for all adapters"""
    
    def __init__(self, async_node):
        super().__init__(async_node.__class__.__name__)
        # rclpy.init()

        self._n = async_node
        self.log = self.get_logger()

        # self._n.params = self._attach_params_dataclass(self._n.params)
        self._loop = asyncio.get_running_loop()

    def __getattr__(self, attr):
        """Lookup unknown attrs on node definition. Effectively extends node def"""
        try:
            return getattr(self._n, attr)
        except AttributeError:
            raise AttributeError(f"Attribute >{attr}< was not found in either [{type(self).__qualname__}] or [{type(self._n).__name__}]")


    def _get_ros_definitions(self):
        from .decorators.import_node import RosImport

        node = self._n
        
        a = [
            (attr, getattr(node, attr))
            for attr in dir(self._n)
            if isinstance(getattr(node, attr), RosDefinition)
        ]

        # Make sure RosImports are processed first
        a.sort(key=lambda b: type(b[1]) != RosImport)

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
            RosParamHandler: self._attach_param_handler,
            RosTimer: self._attach_timer,
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

    def _attach_params_dataclass(self, dataclass):
        if dataclass is None:
            return
        
        # Declare and update all parameters present in the
        # passed dataclass.
        for f in dataclasses.fields(dataclass):
            self.declare_parameter(f.name, f.default)

            # Map dataclass type to a ROS value attribute
            # IE int -> get_parameter_value().integer_value
            getter = getter_map[f.type]

            if getter is None:
                raise RuntimeError(f"No getter for type {f.type}")

            current_val = getattr(
                self.get_parameter(f.name).get_parameter_value(), getter
            )
            setattr(dataclass, f.name, current_val)

        # TODO: Figure out some kind of notification system for continuous updates
        # Important: how to notify updates to application code?
        # self.add_on_set_parameters_callback(self.parameter_callback)

        return dataclass