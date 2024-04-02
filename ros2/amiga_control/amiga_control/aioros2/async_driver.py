import asyncio
import rclpy


class AsyncDriver(rclpy.node.Node):
    """Base class for all adapters"""
    
    def __init__(self, async_node):
        self._n = async_node

        rclpy.init()
        super().__init__(self._n.node_name)
        self._n.params = self._attach_params_dataclass(self._n.params)
        self._loop = asyncio.get_event_loop()
    
    def _get_decorated(self):
        node = self._n
        
        return [
            getattr(node, handler_name)
            for handler_name in dir(self._n)
            if hasattr(getattr(node, handler_name), "_ros_type")
        ]
    
    def _attach_params_dataclass(self, dataclass):
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