import asyncio
import dataclasses
import rclpy
from rclpy.action import ActionServer
import rclpy.node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import ParameterDescriptor, ParameterEvent, ParameterValue

from .async_driver import (
    AsyncDriver,
    dataclass_ros_map,
    ros_type_getter_map,
    dataclass_ros_enum_map,
)
from .result import Result
from .feedback import Feedback

from .decorators import RosDefinition
from .decorators.service import RosService
from .decorators.topic import RosTopic
from .decorators.subscribe import RosSubscription
from .decorators.import_node import RosImport
from .decorators.action import RosAction
from .decorators.timer import RosTimer
from .decorators.params import RosParams
from .decorators.param import RosParam
from .decorators import idl_to_kwargs

# ros2 run amiga_control amiga_control_node --ros-args -p amiga_params_port_canbus:=1234 --remap __node:=test_node --remap __ns:=/test
# ros2 param list
# ros2 param set /test/test_node amiga_params.host 1234.3
# ros2 param get /test/test_node amiga_params_host

# ros2 run amiga_control amiga_control_node --ros-args --remap __node:=test_node --remap __ns:=/test -p "dependant_node_1.name:=dep1" -p "dependant_node_1.ns:=/test"
# ros2 run amiga_control circular_node --ros-args --remap __node:=dep1 --remap __ns:=/test -p "dependant_node_1.name:=test_node" -p "dependant_node_1.ns:=/test"

# https://roboticsbackend.com/rclpy-params-tutorial-get-set-ros2-params-with-python/
# https://roboticsbackend.com/ros2-rclpy-parameter-callback/
class ParamWrapper:
    """Manages a single parameter"""

    def __init__(self, fqpn, dclass_field, node):
        node.log.info(f"Declare parameter >{fqpn}<")

        self.node = node
        self.fqpn = fqpn
        self.field = dclass_field
        self.listeners = []
        self.value = None

        # Initialize parameter
        t = dclass_field.type
        default = dclass_field.default

        if not t in dataclass_ros_map:
            raise TypeError(f"Type >{t}< is not supported by ROS as a parameter.")

        # Declare param
        ros_type = dataclass_ros_map[t]
        descriptor = ParameterDescriptor(type=ros_type)
        node.declare_parameter(fqpn, default, descriptor)

        # Perform initial update
        self.update()

    def update(self):
        p_val = self.node.get_parameter(self.fqpn).get_parameter_value()
        self.update_from_ros_param_value(p_val)

    def set(self, val):
        if not type(val) == self.field.type:
            raise TypeError("WRONG TYPE!")

        ros_type = dataclass_ros_enum_map[self.field.type]
        self.value = val

        return Parameter(self.fqpn, ros_type, val)

    def update_from_ros_param_value(self, pval, trigger_listeners=True):
        getter_fn = ros_type_getter_map[pval.type]
        self.value = getattr(pval, getter_fn)

        if trigger_listeners:
            for listener in self.listeners:
                listener(self.field.name, self.value)

        return self.value


# https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/parameters.md
class ParamsWrapper:
    # https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/parameters.md
    # https://github.com/ros2/demos/blob/rolling/demo_nodes_py/demo_nodes_py/parameters/set_parameters_callback.py

    def __init__(self, params_attr, param_class, ros_node, queue_size=10):
        self.__params_attr = params_attr
        self.__param_class = param_class
        self.__node = ros_node

        self.__params = {}

        self.__node.log.info("Create parameter wrapper")
        self.__node.create_subscription(
            ParameterEvent, "/parameter_events", self.__on_parameter_event, queue_size
        )

        # declare dataclass fields
        for f in dataclasses.fields(self.__param_class):
            if f.name in dir(self):
                raise NameError(
                    f"Parameter name >{f.name}< in >{self.__param_class}< is reserved by aioros2! Please choose another name."
                )

            fqpn = self.__get_fqpn(f.name)
            if isinstance(f.default, RosParam):
                ## TODO
                continue
            else:
                self.__params[fqpn] = ParamWrapper(fqpn, f, ros_node)

        # Attach to param event to handle successful param updates
        # Note: In later versions than foxy, this could be handled natively using
        # self.__ros_node.add_post_set_parameters_callback
        # https://docs.ros.org/en/rolling/Concepts/Basic/About-Parameters.html#

        # TODO: Attach to on_set_parameter to typecheck before accepting param set

    async def update(self):
        def _update():
            for p in self.__params:
                self.__params[p].update()

        await self.__node._loop.run_in_executor(None, _update)

    async def set(self, **kwargs):
        updated_params = []

        for k in kwargs:
            fqpn = self.__get_fqpn(k)

            if not fqpn in self.__params:
                self.__node.log.warn(f">{k}< is not a valid parameter name.")
                continue

            param = self.__params[fqpn]
            val = kwargs[k]

            ros_param = param.set(val)

            updated_params.append(ros_param)

        # Push updates through ROS
        try:
            await self.__node._loop.run_in_executor(
                None, self.__node.set_parameters, updated_params
            )
        except Exception as e:
            print("ERROR SETTING", e)

    def __get_fqpn(self, field_name):
        """Get Fully Qualified Parameter Name - prefixes dclass param names with the instance attribute name to allow
        duplicate parameter names within nodes
        """
        return f"{self.__params_attr}_{field_name}"

    def __on_parameter_event(self, req: ParameterEvent):
        # Check if event is targeting this node
        # print("GOT PARAM EVENT", req)
        if not self.__path_matches_self(req.node):
            return

        for ros_param in req.changed_parameters:
            fqpn = ros_param.name

            if fqpn not in self.__params:
                continue

            param = self.__params[fqpn]

            # Updates parameter value and calls listeners
            param.update_from_ros_param_value(ros_param.value)

    def __getattr__(self, attr):
        fqpn = self.__get_fqpn(attr)
        return self.__params[fqpn].value

    def __path_matches_self(self, path):
        node_namespace = self.__node.get_namespace().lstrip("/")
        fqnp = "/".join([node_namespace, self.__node.get_name()])
        fqnp = "/" + fqnp if not fqnp.startswith("/") else fqnp
        return path == fqnp


class ServerDriver(AsyncDriver, rclpy.node.Node):
    def __init__(self, async_node):
        AsyncDriver.__init__(self, async_node)
        rclpy.node.Node.__init__(self, self.__class__.__name__)

        self.log = self.get_logger()

        # Get ros-asyncio decorated functions to bind.
        for handler in self._get_ros_definitions():
            print(handler)

        self._attach()

    def _process_import(self, attr, imp: RosImport):
        from .client_driver import ClientDriver

        self.log.info("[SERVER] Resolving import")

        # Create a parameter to fully resolve
        node_name_param_name = f"{attr}.name"
        node_namespace_param_name = f"{attr}.ns"

        self.declare_parameter(node_name_param_name, attr)
        self.declare_parameter(node_namespace_param_name, "/")

        node_name = (
            self.get_parameter(node_name_param_name).get_parameter_value().string_value
        )

        node_ns = (
            self.get_parameter(node_namespace_param_name)
            .get_parameter_value()
            .string_value
        )

        if node_name == attr:
            self.log.warn(
                f"Node name for import >{attr}< was not set at "
                f">{node_name_param_name}<. Using default name: >{attr}<"
            )

        if node_ns == "/":
            self.log.warn(
                f"Node namespace for import >{attr}< was not set at "
                f">{node_namespace_param_name}<. Using default namespace: >/<"
            )

        return ClientDriver(imp, self, node_name, node_ns)

    def _attach_service(self, attr, srv_def: RosService):
        self.log.info(f"[SERVER] Attach service @ >{srv_def.namespace}<")

        def cb(req, res):
            print(f"SERVICE HANDLER {srv_def.namespace} START", req)

            result = srv_def.call_handler_sync(self, req, self._loop)

            # Prevent ROS hang if return value is invalid
            if not isinstance(result, Result):
                self.log.warn(
                    f"Service handler @ >{srv_def.namespace}< did not return `result(...)`. "
                    "Expected Result, got {result}"
                )
                result = res
            else:
                result = srv_def.idl.Response(*result.args, **result.kwargs)

            print(f"SERVICE HANDLER {srv_def.namespace} FINISH", result)

            return result

        self.create_service(srv_def.idl, srv_def.namespace, cb)

    def _attach_action(self, attr, d: RosAction):
        self.log.info(f"[SERVER] Attach action >{attr}<")

        def cb(goal):
            print(f"ACTION HANDLER @ {d.namespace} START", goal)

            kwargs = idl_to_kwargs(goal.request)
            gen = d.sever_handler(self, **kwargs)

            result: Feedback | Result = None
            while True:
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        gen.__anext__(),
                        loop=self._loop,
                    ).result()

                    print(f"ACTION HANDLER @ {d.namespace} FEEDBACK", result)

                    if isinstance(result, Feedback):
                        fb = d.idl.Feedback(*result.args, **result.kwargs)
                        goal.publish_feedback(fb)

                    elif isinstance(result, Result):
                        goal.succeed()
                        break

                    else:
                        self.log.error("YIELDED NON-FEEDBACK, NON-RESULT VALUE")

                except StopAsyncIteration:
                    self.log.warn(f"Action >{attr}< returned before yielding `result`")
                    break

            if not isinstance(result, Result):
                self.log.warn(
                    f"Action >{attr}< did not yield `result(...)`. "
                    f"Expected >{d.idl.Result}<, got >{result}<. Aborting action."
                )
                result = d.idl.Result()
            else:
                result = d.idl.Result(*result.args, **result.kwargs)

            print(f"ACTION HANDLER @ {d.namespace} FINISH", result)
            return result

        setattr(
            self,
            f"__{d.sever_handler.__name__}",
            ActionServer(self, d.idl, d.namespace, cb),
        )

    def _attach_subscriber(self, attr, sub: RosSubscription):
        topic = sub.get_topic(self.get_name(), self.get_namespace())

        self.log.info(f"[SERVER] Attach subscriber >{attr}<")
        # print(topic, topic.idl, topic.namespace, topic.qos)

        def cb(msg):
            print(f"SUBSCRIPTION HANDLER {topic.namespace} START", msg)

            kwargs = idl_to_kwargs(msg)

            asyncio.run_coroutine_threadsafe(
                sub.server_handler(self, **kwargs), loop=self._loop
            ).result()

        self.create_subscription(topic.idl, topic.namespace, cb, topic.qos)

    def _attach_publisher(self, attr, topic_def: RosTopic):
        self.log.info(f"Attach publisher @ >{topic_def.namespace}<")

        pub = self.create_publisher(topic_def.idl, topic_def.namespace, topic_def.qos)

        async def _dispatch_pub(*args, **kwargs):
            msg = topic_def.idl(*args, **kwargs)
            await self._loop.run_in_executor(None, pub.publish, msg)

        return _dispatch_pub

    # TODO: Better error handling.
    # ATM raised errors are completely hidden
    def _attach_timer(self, attr, t: RosTimer):
        self.log.info(f"Attach timer >{attr}<")

        def _cb():
            asyncio.run_coroutine_threadsafe(t.server_handler(self), self._loop)

        self.create_timer(t.interval, _cb)

    def _attach_params(self, attr, p: RosParams):
        self.log.info(f"Attach params >{attr}<")

        return ParamsWrapper(attr, p.params_class, self)
