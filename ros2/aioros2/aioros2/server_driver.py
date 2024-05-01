import dataclasses
import inspect
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import traceback

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
from .decorators.start import RosStart

# https://answers.ros.org/question/340600/how-to-get-ros2-parameter-hosted-by-another-node/
# https://roboticsbackend.com/rclpy-params-tutorial-get-set-ros2-params-with-python/
# https://roboticsbackend.com/ros2-rclpy-parameter-callback/
# https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/parameters.md
# https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/parameters.md
# https://github.com/ros2/demos/blob/rolling/demo_nodes_py/demo_nodes_py/parameters/set_parameters_callback.py


class ParamDriver:
    """Manages a single parameter"""

    value = None
    fqpn = None

    __staged_value = None

    def __init__(self, field: dataclasses.Field, fqpn: str) -> None:
        self.fqpn = fqpn
        self.value = field.default

        self.__listeners = []
        self.__ros_type = dataclass_ros_map[field.type]
        self.__ros_enum = dataclass_ros_enum_map[field.type]
        self.__ros_param_getter = ros_type_getter_map[self.__ros_type]

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
        self.__staged_value = getattr(pval, self.__ros_param_getter)

        return SetParametersResult(successful=True)

    def to_ros_param(self, value):
        """Creates a ROS parameter that can be passed to `set_parameters`
        to eventually update this parameter with the passed value"""
        return Parameter(self.fqpn, self.__ros_enum, value)

    def commit(self):
        """Commits the currently staged value"""
        self.value = self.__staged_value

    def get_listeners(self):
        return self.__listeners

    def declare(self, ros_node: "ServerDriver"):
        """Declares this parameter on the passed ros node"""
        ros_node.log_debug(f"Declare parameter >{self.fqpn}<")
        ros_node.declare_parameter(self.fqpn, self.value)


class ParamsDriver:
    """Manages a parameter dataclass"""

    def __init__(self, params_attr, param_class, server_driver: "ServerDriver"):
        self.__params_attr = params_attr
        self.__param_class = param_class
        self.__driver = server_driver

        # Stores parameter drivers by their fqpn (Fully Qualified Parameter Name)
        self.__params = {}

        self.__driver.log_debug("Create parameter wrapper")

        # declare dataclass fields
        for f in dataclasses.fields(self.__param_class):
            if f.name in dir(self):
                raise NameError(
                    f"Parameter name >{f.name}< in >{self.__param_class}< is reserved by aioros2! Please choose another name."
                )

            fqpn = self.__fqpn(f.name)
            param = self.__params[fqpn] = ParamDriver(f, fqpn)

            if isinstance(f.default, RosParam):
                ## TODO: declare to allow descriptors
                continue

            param.declare(self.__driver)

        # Add a callback for when any param is changed to handle local updates
        self.__driver.add_on_set_parameters_callback(self.__parameters_callback)

    def __parameters_callback(self, params):
        """Handles incoming parameter changes on this node"""
        # Filter and pair incoming parameters to only parameters this class manages
        # ros_parameter, parameter_driver
        own_param_pairs = [
            (p, self.__params[p.name]) for p in params if p.name in self.__params
        ]

        # Stage new values
        for in_param, param_driver in own_param_pairs:
            res = param_driver.stage_param(in_param)

            if not res.successful:
                return res

        # If all stages were successful, commit values
        for _, param_driver in own_param_pairs:
            param_driver.commit()

        # Once all values have been commited, run listeners
        all_listeners = [l for _, p in own_param_pairs for l in p.get_listeners()]

        # Remove duplicate listener calls while preserving order
        # https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
        seen = set()
        seen_add = seen.add
        unique_listeners = [x for x in all_listeners if not (x in seen or seen_add(x))]

        # Run unique listeners
        self.__driver.run_coroutine(self.__call_listeners, unique_listeners)

        return SetParametersResult(successful=True)

    async def __call_listeners(self, unique_listeners):
        for l in unique_listeners:
            await l(self.__driver)

    def get_param(self, param_name) -> ParamDriver:
        fqpn = self.__fqpn(param_name)
        return self.get_param_fqpn(fqpn)

    def has_param(self, param_name):
        fqpn = self.__fqpn(param_name)
        return self.has_param_fqpn(fqpn)

    def get_param_fqpn(self, fqpn):
        return self.__params[fqpn]

    def has_param_fqpn(self, fqpn):
        return fqpn in self.__params

    def add_change_listener(self, param_name, listener):
        """Adds a listener function to the specified parameter"""
        self.get_param(param_name).add_listener(listener)

    async def set(self, **kwargs):
        """Sets parameters. Takes many kwargs in the form `parameter_name=new_value` and executes a ROS mutation.
        Parameters are changed atomically in the same call, not once at a time"""
        updated_params = []

        for param_name in kwargs:
            new_value = kwargs[param_name]
            if not self.has_param(param_name):
                self.__driver.log_warn(f">{param_name}< is not a valid parameter name.")
                continue

            param = self.get_param(param_name)

            updated_params.append(param.to_ros_param(new_value))

        # Push updates through ROS
        await self.__driver.run_executor(self.__driver.set_parameters, updated_params)

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

        self._attach()

    def _process_import(self, attr, ros_import: RosImport):
        from .client_driver import ClientDriver

        self.log_debug("[SERVER] Resolving import")

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
            self.log_warn(
                f"Node name for import >{attr}< was not set at "
                f">{node_name_param_name}<. Using default name: >{attr}<"
            )

        if node_ns == "/":
            self.log_warn(
                f"Node namespace for import >{attr}< was not set at "
                f">{node_namespace_param_name}<. Using default namespace: >/<"
            )

        return ClientDriver(ros_import, self, node_name, node_ns)

    def _attach_service(self, attr, ros_service: RosService):
        """Attaches a service"""
        self.log_debug(f"[SERVER] Attach service >{attr}< @ >{ros_service.path}<")

        # Will be called from MultiThreadedExecutor
        def cb(req, res):
            # Call handler function
            kwargs = idl_to_kwargs(req)
            result = self.run_coroutine(ros_service.handler, self, **kwargs).result()

            if isinstance(result, Result):
                result = ros_service.idl.Response(*result.args, **result.kwargs)

            else:  # Prevent ROS hang if return value is invalid by returning default
                self._logger.warn(
                    f"Service handler >{attr}< @ >{ros_service.path}< did not return `result(...)`. "
                    f"Expected Result, got {result}"
                )
                result = res

            return result

        self.create_service(ros_service.idl, ros_service.path, cb)

    def _attach_action(self, attr, ros_action: RosAction):
        self.log_debug(f"[SERVER] Attach action >{attr}<")

        def cb(goal):
            kwargs = idl_to_kwargs(goal.request)
            gen = ros_action.handler(self, **kwargs)

            result: Feedback | Result = None
            while True:
                try:
                    result = self.run_coroutine(gen.__anext__()).result()

                    if isinstance(result, Feedback):
                        fb = ros_action.idl.Feedback(*result.args, **result.kwargs)
                        goal.publish_feedback(fb)

                    elif isinstance(result, Result):
                        goal.succeed()
                        break

                    else:
                        self.log_error("YIELDED NON-FEEDBACK, NON-RESULT VALUE")

                except StopAsyncIteration:
                    self.log_warn(f"Action >{attr}< returned before yielding `result`")
                    break

            if isinstance(result, Result):
                result = ros_action.idl.Result(*result.args, **result.kwargs)
            else:
                self.log_warn(
                    f"Action >{attr}< did not yield `result(...)`. "
                    f"Expected >{ros_action.idl.Result}<, got >{result}<. Aborting action."
                )
                result = ros_action.idl.Result()

            print(f"ACTION HANDLER @ {ros_action.path} FINISH", result)
            return result

        setattr(
            self,
            f"__{ros_action.handler.__name__}",
            ActionServer(self, ros_action.idl, ros_action.path, cb),
        )

        return ros_action.handler

    def _attach_subscriber(self, attr, ros_sub: RosSubscription):
        fqt = ros_sub.get_fqt(self.get_name(), self.get_namespace())

        self.log_debug(f"[SERVER] Attach subscriber >{attr}<")

        def cb(msg):
            kwargs = idl_to_kwargs(msg)
            self.run_coroutine(ros_sub.handler, self, **kwargs)

        self.create_subscription(fqt.idl, fqt.path, cb, fqt.qos)

    def _attach_publisher(self, attr, ros_topic: RosTopic):
        self.log_debug(f"Attach publisher {attr} @ >{ros_topic.path}<")

        pub = self.create_publisher(ros_topic.idl, ros_topic.path, ros_topic.qos)

        async def _dispatch_pub(*args, **kwargs):
            msg = ros_topic.idl(*args, **kwargs)
            await self.run_executor(pub.publish, msg)

        return _dispatch_pub

    # TODO: Better error handling.
    # ATM raised errors are completely hidden
    def _attach_timer(self, attr, ros_timer: RosTimer):
        self.log_debug(f"Attach timer >{attr}<")

        def _cb():
            try:
                self.run_coroutine(ros_timer.server_handler, self)
            except Exception:
                self.log_error(traceback.format_exc())

        self.create_timer(ros_timer.interval, _cb)
        return ros_timer.server_handler

    def _attach_params(self, attr, ros_params: RosParams):
        self.log_debug(f"Attach params >{attr}<")
        return ParamsDriver(attr, ros_params.params_class, self)

    def _attach_param_subscription(self, attr, ros_param_sub: RosParamSubscription):
        # For each reference, find its definition's corrosponding attribute name
        # so that we can reference its instantiated version through getattr.
        for ref in ros_param_sub.references:
            for member in inspect.getmembers(self._n):
                if member[1] == ref.params_def:
                    attr = member[0]
                    break

            if not attr:
                # TODO: improve this error message
                raise AttributeError("NOT FOUND")

            # Add a listener to the corrosponding ParamsWrapper, param
            params_wrapper: ParamsDriver = getattr(self, attr)
            params_wrapper.add_change_listener(ref.param_name, ros_param_sub.handler)

        # Allow handling function to be directly called
        return ros_param_sub.handler

    def _process_start(self, attr, ros_start: RosStart):
        self.log_debug(f"Process start >{attr}<")
        self.run_coroutine(ros_start.server_handler, self)
        return ros_start.server_handler
