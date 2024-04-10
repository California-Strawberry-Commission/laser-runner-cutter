import asyncio
import dataclasses
import rclpy
from .decorators.self import Self
from .decorators import RosDefinition

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