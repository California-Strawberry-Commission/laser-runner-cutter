from queue import Empty, SimpleQueue
from rcl_interfaces.srv import DescribeParameters, GetParameters, GetParameterTypes
import rclpy
import rclpy.logging
import rclpy.node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.expand_topic_name import expand_topic_name

from . import server_driver
from .async_driver import AsyncDriver
from .decorators.action import RosAction
from .decorators.import_node import RosImport
from .decorators.params import RosParams
from .decorators.service import RosService
from .decorators.subscribe import RosSubscription
from .decorators.timer import RosTimer
from .decorators.topic import RosTopic
from .decorators.start import RosStart

# Extend RosTopic for downstream references
class CachedSubscription(RosTopic):
    def __init__(self, topic: RosTopic, client: "ClientDriver"):
        super().__init__(topic.path, topic.idl, topic.qos)
        self.node = client
        self.value = None 

        fqt = expand_topic_name(topic.path, client._node_name, client._node_namespace)

        client.log_debug(f"[SERVER] Attach topic cache >{fqt}<")

        def cb(msg):
            self.value = msg

        client._node.create_subscription(topic.idl, fqt, cb, topic.qos)
        
        
class AsyncActionClient:
    _action_complete = False
    result = None

    def __init__(self, gen, loop, idl):
        if gen is None:
            self._action_complete = True

        self._gen = gen
        self._loop = loop
        self._idl = idl

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._action_complete:
            raise StopAsyncIteration

        def _wrap():
            try:
                return self._gen.__next__()
            except StopIteration:
                raise StopAsyncIteration

        val = await self._loop.run_in_executor(None, _wrap)

        if isinstance(val, self._idl.Result):
            self.result = val
            raise StopAsyncIteration
        elif isinstance(val, self._idl.Impl.FeedbackMessage):
            return val.feedback



class ClientDriver(AsyncDriver):
    """
    Generates an interface to communicate with an imported node from a server driver.
    """

    def __init__(
        self, node_def, server_node, node_name, node_namespace=None, logger=None
    ):
        self._node: "server_driver.ServerDriver" = server_node


        if logger is None:
            logger = rclpy.logging.get_logger(self._get_logger_name(node_name, node_namespace))

        super().__init__(node_def, logger, node_name, node_namespace)

        self._callback_group = ReentrantCallbackGroup()

        self._attach()

    def _get_logger_name(self, name, ns):
        """Returns a pretty name for this client's logger"""
        if name == "/":
            return f"{name}-client"
        else:
            ns = ns.lstrip("/")
            return f"{ns}.{name}-client"

    def _process_import(self, attr, ros_import: RosImport):
        self.log_debug("Processing client import")

        # Get node name and namespace for the import
        node_name_param_name = f"{attr}.name"
        node_namespace_param_name = f"{attr}.ns"

        # Lookup the import parameters on the remote server node
        # Need to use param api because these params are not local to this server node.
        fqt = expand_topic_name("~/get_parameters", self._node_name, self._node_namespace)

        param_cli = self._node.create_client(GetParameters, fqt)

        while not param_cli.wait_for_service(timeout_sec=1.0):
            self.log_debug(f'>{fqt}< service not available.')

        req = GetParameters.Request(names=[node_name_param_name, node_namespace_param_name])

        # Block asyncio to perform this service call - initialization is sync so this
        # call needs to be too.
        res = param_cli.call_async(req)
        while rclpy.ok():
            rclpy.spin_once(self._node)
            if res.done():
                res = res.result()
                break
        
        import_name_param, import_ns_param = res.values

        # The configured import name and namespace!
        import_name = import_name_param.string_value
        import_ns = import_ns_param.string_value or "/"

        if import_name == '':
            self.log_warn(f"Could not complete import for >{attr}< - "
                          f"Service was found, but params >{node_name_param_name}< "
                          f"or >{node_namespace_param_name}< were not set.")

        # If the referenced node is the same as the serverDriver (circular reference)
        # DON'T create a client driver. Return the server driver.
        if import_name == self._node._node_name and import_ns == self._node._node_namespace:
            return self._node

        node_def = ros_import.resolve()

        # Create a new clientdriver for this node
        return ClientDriver(node_def, self._node, import_name, import_ns)

    def _attach_publisher(self, attr, topic: RosTopic):
        topic.node = self # Set topic node in definition so other attachers know about it.
        
        return CachedSubscription(topic, self)

    def _attach_timer(self, attr, ros_timer: RosTimer):
        # Unused on clients
        return None

    def _attach_params(self, attr, ros_params: RosParams):
        # Unused on clients
        return None

    def _process_start(self, attr, ros_start: RosStart):
        # Unused on clients
        return None

    def _attach_subscriber(self, attr, ros_sub: RosSubscription):
        # TODO: This doesn't work as expected for 2nd order imports
        topic = ros_sub.get_fqt()

        # Creates a publisher for this channel
        self.log_debug(f"[CLIENT] Attach subscriber publisher @ >{topic.path}<")

        pub = self._node.create_publisher(topic.idl, topic.path, topic.qos)

        async def _dispatch_pub(*args, **kwargs):
            msg = topic.idl(*args, **kwargs)
            await self._node.run_executor(pub.publish, msg)

        return _dispatch_pub

    def _attach_action(self, attr, ros_action: RosAction):
        self.log_debug(f"[CLIENT] Attach action >{attr}<")

        client = ActionClient(
            self._node,
            ros_action.idl,
            self._resolve_path(ros_action.path),
            callback_group=self._callback_group,
        )

        def _impl(*args, **kwargs):
            goal = ros_action.idl.Goal(*args, **kwargs)
            generator = self._dispatch_action_req(client, goal)
            return AsyncActionClient(generator, self._node._loop, ros_action.idl)

        return _impl

    # TODO: Gracefully handle action rejections
    def _dispatch_action_req(self, client, request):
        """Dispatches a ROS action request and yields feedback as it is received
        side note: wow is the ROS api for this horrible or what???
        """
        # https://docs.ros.org/en/foxy/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html

        action_complete = False
        feedback_queue = SimpleQueue()
        result = None

        # Wait for desired service to become available
        if not client.wait_for_server(timeout_sec=2.0):
            self._logger.error("Action server not available")
            return None

        # Create method-local hooks for callbacks
        def on_feedback(fb):
            nonlocal feedback_queue
            feedback_queue.put(fb)

        def on_result(future):
            nonlocal action_complete, result
            result = future.result().result
            action_complete = True

        def on_initial_action_response(future):
            """Entrypoint for initial action response"""
            nonlocal action_complete

            goal_handle = future.result()
            if not goal_handle.accepted:
                self.log_warn("Goal rejected")
                action_complete = True
                return

            # Attach goal callbacks
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(on_result)

        # Request action w/ goal
        fut = client.send_goal_async(request, feedback_callback=on_feedback)
        fut.add_done_callback(on_initial_action_response)

        # Spin to win
        while not action_complete:
            # rclpy.spin_once(self)
            try:
                yield feedback_queue.get_nowait()
            except Empty:
                pass

        # TODO: yield unyielded feedback?

        yield result
        raise StopAsyncIteration()

    def _attach_service(self, attr, ros_service: RosService):
        self.log_debug(f"[CLIENT] Attach service >{attr}< @ >{ros_service.path}<")

        ros_client = self._node.create_client(
            ros_service.idl,
            self._resolve_path(ros_service.path),
            callback_group=self._callback_group,
        )

        async def _impl(*args, **kwargs):
            request = ros_service.idl.Request(*args, **kwargs)
            if not ros_client.wait_for_service(timeout_sec=2.0):
                self._logger.error("Service not available")
                return None
            return await ros_client.call_async(request)

        return _impl

    # Resolves a path into a fully resolved path based on this client's
    # fully qualified node path
    # https://design.ros2.org/articles/topic_and_service_names.html
    def _resolve_path(self, path: str):
        return expand_topic_name(path, self._node_name, self._node_namespace)
