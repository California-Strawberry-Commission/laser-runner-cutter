import asyncio
import rclpy
import rclpy.node
from amiga_control_interfaces.srv import SetTwist
from enum import Enum
import dataclasses

# TODO: flesh this out.
getter_map = {
    str: "string_value",
    int: "integer_value",
}

# Future note: dataclass requires type annotations to work
@dataclasses.dataclass
class Params:
    host: str = "127.0.0.1"
    port_canbus: int = 6001



class AmigaControlNode(rclpy.node.Node):
    def register_params_dataclass(self, dclass):
        # Declare and update all parameters present in the
        # passed dataclass.
        for f in dataclasses.fields(dclass):
            self.declare_parameter(f.name, f.default)

            # Map dataclass type to a ROS value attribute
            # IE int -> get_parameter_value().integer_value
            getter = getter_map[f.type]

            if getter is None:
                raise RuntimeError(f"No getter for type {f.type}")

            current_val = getattr(
                self.get_parameter(f.name).get_parameter_value(), getter
            )
            setattr(dclass, f.name, current_val)

        # TODO: Figure out some kind of notification system ://
        # self.add_on_set_parameters_callback(self.parameter_callback)

        return dclass
    
    def __init__(self):
        super().__init__("amiga_control_node")

        self.params = self.register_params_dataclass(Params())

        # Init services
        self.create_service(SetTwist, "~/set_twist", self.cb_set_twist)

    def cb_set_twist(self, req, res):
        print(req.twist)

        res.success = True
        return res


async def ros_loop(node):
    print("Ros node starting up")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0)
        await asyncio.sleep(1e-4)


def main():
    rclpy.init()   
    node = AmigaControlNode()
    
    future = asyncio.wait([ros_loop(node)])
    asyncio.get_event_loop().run_until_complete(future)

if __name__ == "__main__":
    main()
