import asyncio
import dataclasses
import re
from typing import Optional
import rclpy
import rclpy.node
import atexit
import inspect

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process


def to_camel_case(snake_str):
    # https://stackoverflow.com/questions/19053707/converting-snake-case-to-lower-camel-case-lowercamelcase
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_snake(camel_str):
    camel_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()


async def ros_spin_nodes(nodes):
    print("Ros starting up...")

    # TODO: Tune thread counts. Might be limitations - not sure what ROS behavior when running out.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=10)

    for n in nodes:
        executor.add_node(n)

    print("Ros event loop running!")
    while rclpy.ok():
        executor.spin_once(timeout_sec=0)
        await asyncio.sleep(1e-4)


def serve_nodes(*nodes):
    rclpy.init()

    servers = [n.server() for n in nodes]
    tasks = [task for s in servers for task in s.tasks()]

    tasks = asyncio.wait(tasks + [ros_spin_nodes(servers)])
    asyncio.get_event_loop().run_until_complete(tasks)


# TODO: Check types, check annotated return value if exists.
def _check_service_function_signature(fn, srv):
    fn_inspection = inspect.signature(fn)
    print(fn_inspection)
    fn_dict = fn_inspection.parameters
    fn_params = set(fn_dict)
    fn_params.discard("self")

    idl_dict = srv.Request.get_fields_and_field_types()
    idl_params = set(idl_dict.keys())

    if fn_params != idl_params:
        raise RuntimeError(
            "Service handler parameters do not match those in the IDL format! "
            "Make sure that the function parameter names match those in the IDL!\n"
            f"Handler: {fn.__name__} -> \t{fn_params if len(fn_params) else 'NO ARGUMENTS'}\n"
            f"    IDL: {srv.__name__} -> \t{idl_params}"
        )


def decorate_handler(handler, ros_type, ros_namespace=None, ros_idl=None):
    handler._ros_type = ros_type
    handler._ros_namespace = ros_namespace
    handler._ros_idl = ros_idl


# https://stackoverflow.com/questions/11731136/class-method-decorator-with-self-arguments
def service(namespace, srv_idl):
    def _service(fn):
        _check_service_function_signature(fn, srv_idl)
        decorate_handler(fn, "service", ros_idl=srv_idl, ros_namespace=namespace)
        return fn

    return _service


def action(namespace, act_idl):
    def _action(fn):
        _check_service_function_signature(fn, act_idl)
        decorate_handler(fn, "action", ros_idl=act_idl, ros_namespace=namespace)

        return fn

    return _action


def timer(interval):
    def _timer(fn):
        decorate_handler(fn, "timer")
        return fn

    return _timer


def param(dataclass_param):
    def _param(fn):
        decorate_handler(fn, "param_event")
        return fn

    return _param


def rosnode(params_class):
    def _rosnode(cls):
        original_init = cls.__init__

        def wrapped_init(self, node_name=to_snake(cls.__name__), params=params_class()):
            original_init(self, node_name, params)

        cls.__init__ = wrapped_init

        return cls

    return _rosnode


class AsyncClientNode(rclpy.node.Node):
    def __init__(self, async_node):
        super().__init__(async_node.node_name + "_client")
        self._n = async_node
        self._n.params = self._attach_params_dataclass(self._n.params)
        self._loop = asyncio.get_event_loop()

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


class AsyncServerNode(rclpy.node.Node):
    def __init__(self, async_node):
        super().__init__(async_node.node_name)
        self._n = async_node
        self._n.params = self._attach_params_dataclass(self._n.params)
        self._loop = asyncio.get_event_loop()

        # Get ros-asyncio decorated functions to bind.
        for handler in self._n._get_ros_handlers():
            print(handler)

    def tasks(self):
        tasks = []

        for handler in self._n._get_ros_handlers():
            if handler._ros_type == "service":
                self._attach_service(handler)
            else:
                pass
                # raise RuntimeError("Unsupported type")

        return tasks

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

    def _attach_service(self, service_handler):
        def cb(req, res):
            print("SERVICE HANDLER START", req)
            
            msg_keys = req.get_fields_and_field_types().keys()
            
            kwargs = { k:getattr(req, k) for k in msg_keys }
            
            result = asyncio.run_coroutine_threadsafe(service_handler(**kwargs), loop=self._loop).result()
            
            print("SERVICE HANDLER RUN", result)
            res.success = result
            return res

        self.create_service(
            service_handler._ros_idl, service_handler._ros_namespace, cb
        )


# TODO: Wrap dataclass w/ a setattr trap to trigger notifications
ros_params = dataclasses.dataclass


# Generates a class with passed default values.
# Syntax sugar to remove need to override __init__
class AsyncNode:
    def __init__(
        self,
        node_name: str,
        params: dataclasses.dataclass,
    ):
        self.__name__ = to_camel_case(node_name)
        self.node_name = node_name
        self.params = params

    def _get_ros_handlers(self):
        return [
            getattr(self, d)
            for d in dir(self)
            if hasattr(getattr(self, d), "_ros_type")
        ]

    def server(self):
        return AsyncServerNode(self)

    def client(self):
        return AsyncClientNode(self)


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
