import asyncio
import dataclasses
import inspect
import rclpy
from rclpy.node import Node
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
from .decorators.param_subscription import RosParamSubscription

# ros2 run amiga_control amiga_control_node --ros-args -p amiga_params_port_canbus:=1234 --remap __node:=test_node --remap __ns:=/test
# ros2 param list
# ros2 param set /test/test_node amiga_params.host 1234.3
# ros2 param get /test/test_node amiga_params_host

# ros2 run amiga_control amiga_control_node --ros-args --remap __node:=acn --remap __ns:=/ns1 -p "dependant_node_1.name:=circ" -p "dependant_node_1.ns:=/ns2"
# ros2 run amiga_control circular_node --ros-args --remap __node:=circ --remap __ns:=/ns2 -p "dependant_node_1.name:=acn" -p "dependant_node_1.ns:=/ns1"

# https://answers.ros.org/question/340600/how-to-get-ros2-parameter-hosted-by-another-node/
# https://roboticsbackend.com/rclpy-params-tutorial-get-set-ros2-params-with-python/
# https://roboticsbackend.com/ros2-rclpy-parameter-callback/
# https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/parameters.md
# https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/parameters.md
# https://github.com/ros2/demos/blob/rolling/demo_nodes_py/demo_nodes_py/parameters/set_parameters_callback.py


class Param:
    value = None
    fqpn = None

    __staged_value = None

    def __init__(self, field: dataclasses.Field, fqpn: str) -> None:
        self.fqpn = fqpn
        self.value = field.default

        self.__field = field
        self.__listeners = []
        self.__ros_type = dataclass_ros_map[field.type]
        self.__ros_enum = dataclass_ros_enum_map[field.type]
        self.__param_getter = ros_type_getter_map[self.__ros_type]

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def stage_param(self, param: Parameter):
        pval = param.get_parameter_value()

        # Check incoming type
        if pval.type != self.__ros_type:
            return SetParametersResult(
                successful=False,
                reason=f"Incorrect type for >{self.fqpn}<. Got >{pval.type}<, expected >{self.__ros_type}<",
            )

        # Stage value to set later if all grouped param sets pass
        self.__staged_value = getattr(pval, self.__param_getter)

        return SetParametersResult(successful=True)

    def create_ros_parameter_setter(self, value):
        return Parameter(self.fqpn, self.__ros_enum, value)

    def commit(self):
        # TODO: Call listeners!
        self.value = self.__staged_value

    def get_listeners(self):
        return self.__listeners

    def declare(self, ros_node):
        ros_node.log.info(f"Declare parameter >{self.fqpn}<")
        ros_node.declare_parameter(self.fqpn, self.value)


class ParamsWrapper:
    def __init__(self, params_attr, param_class, server_driver):
        self.__params_attr = params_attr
        self.__param_class = param_class
        self.__driver = server_driver
        self.__params = {}

        self.__driver.log.info("Create parameter wrapper")

        # declare dataclass fields
        for f in dataclasses.fields(self.__param_class):
            if f.name in dir(self):
                raise NameError(
                    f"Parameter name >{f.name}< in >{self.__param_class}< is reserved by aioros2! Please choose another name."
                )

            fqpn = self.__fqpn(f.name)
            param = self.__params[fqpn] = Param(f, fqpn)

            if isinstance(f.default, RosParam):
                ## TODO: declare
                continue

            param.declare(self.__driver)

        # Note: In later versions than foxy, this could be handled using
        # self.__ros_node.add_post_set_parameters_callback
        # https://docs.ros.org/en/rolling/Concepts/Basic/About-Parameters.html#

        self.__driver.add_on_set_parameters_callback(self.__parameters_callback)

    def __parameters_callback(self, params):
        # Filter and pair incoming parameters to parameters this
        # class manages
        own_param_pairs = [
            (p, self.__params[p.name]) for p in params if p.name in self.__params
        ]

        # Stage new values
        for in_param, param in own_param_pairs:
            res = param.stage_param(in_param)

            if not res.successful:
                return res

        # If all stages were successful, commit values
        for _, param in own_param_pairs:
            param.commit()

        # Once all values have been commited, run any listeners
        all_listeners = [l for _, p in own_param_pairs for l in p.get_listeners()]
        print(all_listeners)

        # https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
        seen = set()
        seen_add = seen.add
        unique_listeners = [x for x in all_listeners if not (x in seen or seen_add(x))]

        asyncio.run_coroutine_threadsafe(
            self.__call_listeners(unique_listeners), loop=self.__driver._loop
        )

        return SetParametersResult(successful=True)
    
    async def __call_listeners(self, unique_listeners):
        for l in unique_listeners:
            await l(self.__driver)

    def add_change_listener(self, param_name, listener):
        fqpn = self.__fqpn(param_name)
        param = self.__params[fqpn]
        param.add_listener(listener)

    async def set(self, **kwargs):
        updated_params = []

        for k in kwargs:
            fqpn = self.__fqpn(k)

            if not fqpn in self.__params:
                self.__driver.log.warn(f">{k}< is not a valid parameter name.")
                continue

            param = self.__params[fqpn]

            updated_params.append(param.create_ros_parameter_setter(kwargs[k]))

        # Push updates through ROS
        try:
            await self.__driver._loop.run_in_executor(
                None, self.__driver.set_parameters, updated_params
            )
        except Exception as e:
            print("ERROR SETTING", e)

    def __fqpn(self, field_name):
        """Get Fully Qualified Parameter Name - prefixes dclass param names with the instance attribute name to allow
        duplicate parameter names within nodes
        """
        return f"{self.__params_attr}.{field_name}"

    def __getattr__(self, attr):
        fqpn = self.__fqpn(attr)
        return self.__params[fqpn].value


class ServerDriver(AsyncDriver, Node):
    def __init__(self, async_node):
        Node.__init__(self, self.__class__.__name__)
        AsyncDriver.__init__(self, async_node, self.get_logger())

        self.log = self.get_logger()

        # Get ros-asyncio decorated functions to bind.
        for handler in self._get_ros_definitions():
            print(handler)

        self._attach()

    def _process_import(self, attr, imp: RosImport):
        from .client_driver import ClientDriver

        self.log.info("[SERVER] Resolving import")

        # TODO: extract naming logic somewhere else. This is duplicated
        # in launch_driver._process_imports

        # Create parameters to pass import name & namespace
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
        return t.server_handler

    def _attach_params(self, attr, p: RosParams):
        self.log.info(f"Attach params >{attr}<")
        return ParamsWrapper(attr, p.params_class, self)

    def _attach_param_subscription(self, attr, p: RosParamSubscription):
        print(p.references, p.handler)
        # For each reference, find its definition's corrosponding attribute name
        # so that we can reference its instantiated version through getattr.
        for ref in p.references:
            for member in inspect.getmembers(self._n):
                if member[1] == ref.params_def:
                    attr = member[0]
                    break

            if not attr:
                # TODO: improve this error message
                raise AttributeError("NOT FOUND")

            # Add a listener to the corrosponding ParamsWrapper, param
            params_wrapper: ParamsWrapper = getattr(self, attr)
            params_wrapper.add_change_listener(ref.param_name, p.handler)

        # Allow handling function to be directly called
        return p.handler
