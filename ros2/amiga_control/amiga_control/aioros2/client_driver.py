import rclpy
import rclpy.node
from .async_driver import AsyncDriver
from .decorators._decorators import ros_type_e

class ClientDriver(AsyncDriver):
    def __init__(self, async_node):
        rclpy.init()
        super().__init__(async_node)

        self._create_impl()

    def _create_impl(self):
        attachers = {
            ros_type_e.SERVICE: self._create_service_impl
        }

        for handler in self._get_decorated():
            try:
                attachers[handler._ros_type](handler)
            except KeyError:
                self.log.warn(f"Could not create client implementation for >{handler.__name__}< because type >{handler._ros_type}< is unknown.")

    
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
        
