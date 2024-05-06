from queue import Empty, SimpleQueue

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


# TODO: Receive fully qualified node path here.
class ClientDriver(AsyncDriver):
    def __init__(
        self,
        node_def,
        server_node,
        node_name,
        node_namespace=None,
        logger=None,
        monitor_performance=False,
    ):
        self._node: "server_driver.ServerDriver" = server_node
        self._node_name = node_name
        self._node_namespace = node_namespace if node_namespace is not None else "/"

        if logger is None:
            logger = rclpy.logging.get_logger(self._get_logger_name())

        super().__init__(node_def, logger, monitor_performance)

        self._callback_group = ReentrantCallbackGroup()

        self._attach()

    def _get_logger_name(self):
        """Returns a pretty name for this client's logger"""
        if self._node_namespace == "/":
            return f"{self._node_name}-client"
        else:
            ns = self._node_namespace.lstrip("/")
            return f"{ns}.{self._node_name}-client"

    def _process_import(self, attr, ros_import: RosImport):
        self.log_debug("[CLIENT] Process import")
        # DON'T create a client from a client.
        # Resolve import to raw node definition
        # TODO: Change this to properly resolve 2nd order imports
        return ros_import.resolve()

    def _attach_publisher(self, attr, ros_topic: RosTopic):
        # Don't create topic publisher on clients, but keep definition populated
        return ros_topic

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
        topic = ros_sub.get_fqt(self._node_name, self._node_namespace)

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
