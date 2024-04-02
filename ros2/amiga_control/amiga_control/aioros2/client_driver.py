class ClientDriver(rclpy.node.Node):
    def __init__(self, async_node):
        rclpy.init()
        super().__init__(async_node.node_name)
        self._n = async_node
        self._n.params = self._attach_params_dataclass(self._n.params)
        self._loop = asyncio.get_event_loop()
        
        self._create_impl()

    def _create_impl(self):
        for fn in self._n._get_ros_handlers():
            if fn._ros_type == "service":
                self._create_service_impl(fn)
    
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