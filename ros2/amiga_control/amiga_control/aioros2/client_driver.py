import asyncio
import rclpy
import rclpy.node
from .async_driver import AsyncDriver
from .decorators._decorators import ros_type_e
from rclpy.action import ActionClient
from queue import Empty, SimpleQueue

from rclpy.action.client import ClientGoalHandle

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
    def __init__(self, async_node):
        rclpy.init()
        super().__init__(async_node)

        self._create_impl()

    def _create_impl(self):
        attachers = {
            ros_type_e.SERVICE: self._create_service_impl,
            ros_type_e.ACTION: self._create_action_impl,
            ros_type_e.TOPIC_SUBSCRIBER: self._create_subscriber_publisher_impl
        }

        for handler in self._get_decorated():
            try:
                attachers[handler._ros_type](handler)
            except KeyError:
                self.log.warn(f"Could not create client implementation for >{handler.__name__}< because type >{handler._ros_type}< is unknown.")

    def _create_action_impl(self, handler):
        idl = handler._ros_idl
        
        def _impl(*args, **kwargs):
            goal = handler._ros_idl.Goal(*args, **kwargs)
            generator=self._dispatch_action_req(_impl, goal)
            return AsyncActionClient(generator, self._loop, handler._ros_idl)
            
        
        _impl._ros_client = ActionClient(self, handler._ros_idl, handler._ros_namespace)
        setattr(self, handler.__name__, _impl)
    
    # TODO: Gracefully handle action rejections
    def _dispatch_action_req(self, impl, request):
        """Dispatches a ROS action request and yields feedback as it is received
        side note: wow is the ROS api for this horrible or what???
        """
        # https://docs.ros.org/en/foxy/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html
        
        action_complete = False
        feedback_queue = SimpleQueue()
        result = None
        
        cli = impl._ros_client
        
        # Wait for desired service to become available
        if not impl._ros_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server not available")
            return None
        
        # Create method-local hooks for callbacks
        def on_feedback(fb):
            nonlocal feedback_queue
            feedback_queue.put(fb)
            # print("FEEDBACK", fb)
        
        def on_result(future):
            nonlocal action_complete, result
            result = future.result().result
            # print("DONE", result)
            action_complete = True
            
        def on_action_response(future):
            """Entrypoint for initial action response"""
            nonlocal action_complete
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.log.warn('Goal rejected')
                action_complete = True
                return
            
            # Attach goal callbacks
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(on_result)
        
        # Request action w/ goal
        fut = cli.send_goal_async(request, feedback_callback=on_feedback)
        fut.add_done_callback(on_action_response)
        
        # Spin to win
        while not action_complete:
            rclpy.spin_once(self)
            try:
                yield feedback_queue.get_nowait()
            except Empty:
                pass
                
        # TODO: yield unyielded feedback?
        
        yield result
        raise StopAsyncIteration()
    
    def _create_service_impl(self, handler):
        async def _impl(*args, **kwargs):
            req = handler._ros_idl.Request(*args, **kwargs)
            return await self._loop.run_in_executor(None, self._dispatch_service_req, _impl, req)
        
        _impl._ros_client = self.create_client(handler._ros_idl, handler._ros_namespace)
        setattr(self, handler.__name__, _impl)
        

    def _dispatch_service_req(self, impl, request):
        cli = impl._ros_client
        
        if not impl._ros_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Service not available")
            return None
        
        fut = cli.call_async(request)
        rclpy.spin_until_future_complete(self, fut)
        return fut.result()
        
    def _create_subscriber_publisher_impl(self, handler):
        self.log.info(f"Attach Subscriber publisher >{handler._ros_namespace}<")

        idl = handler._ros_idl
        namespace = handler._ros_namespace
        qos = handler._ros_qos

        print(idl, namespace, qos)
        
        pub = self.create_publisher(idl, namespace, qos)

        def _dispatch(*args, **kwargs):
            msg = idl(*args, **kwargs)
            pub.publish(msg)

        setattr(self, handler.__name__, _dispatch)