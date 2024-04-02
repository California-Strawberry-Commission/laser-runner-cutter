import asyncio
import rclpy

class ServerDriver(rclpy.node.Node):
    def __init__(self, async_node):
        super().__init__(async_node.node_name)
        self.log = self.get_logger()
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
            elif handler._ros_type == "action":
                self._attach_action(handler)
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
        self.log.info(f"Attach service {service_handler.__name__}")

        handler_name = service_handler.__name__
        idl = service_handler._ros_idl
        namespace = service_handler._ros_namespace

        def cb(req, res):
            print(f"SERVICE HANDLER {namespace} START", req)

            msg_keys = req.get_fields_and_field_types().keys()

            kwargs = {k: getattr(req, k) for k in msg_keys}

            result = asyncio.run_coroutine_threadsafe(
                service_handler(respond=idl.Response, **kwargs), loop=self._loop
            ).result()

            # Prevent ROS hang if return value is invalid
            if not isinstance(result, idl.Response):
                self.log.warn(
                    f"Service handler {handler_name} did not return `respond(...)`. "
                    "Expected {idl.Response}, got {result}"
                )
                result = res

            print(f"SERVICE HANDLER {namespace} FINISH", result)

            return result

        self.create_service(idl, namespace, cb)

    def _attach_action(self, action_handler):
        self.log.info(f"Attach action {action_handler.__name__}")
        handler_name = action_handler.__name__
        idl = action_handler._ros_idl
        namespace = action_handler._ros_namespace

        def cb(goal):
            print(f"ACTION HANDLER {namespace} START", goal)
            req = goal.request
            goal_keys = req.get_fields_and_field_types().keys()

            kwargs = {k: getattr(req, k) for k in goal_keys}

            gen = action_handler(respond=idl.Result, feedback=idl.Feedback, **kwargs)

            result = None
            while True:
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        gen.__anext__(),
                        loop=self._loop,
                    ).result()

                    print(f"ACTION HANDLER {namespace} FEEDBACK", result)

                    if isinstance(result, idl.Feedback):
                        goal.publish_feedback(result)
                    elif isinstance(result, idl.Result):
                        goal.succeed()
                        break
                    else:
                        self.log.error("WARN: YIELDED NON-FEEDBACK, NON-RESPOND VALUE")
                except StopAsyncIteration:
                    break

            # Prevent ROS hang if return value is invalid
            if not isinstance(result, idl.Result):
                self.log.warn(
                    f"Service handler {handler_name} did not yield `respond(...)`. "
                    f"Expected {idl.Result}, got {result}. Aborting action."
                )
                result = idl.Result()

            print(f"ACTION HANDLER {namespace} FINISH", result)
            return result

        setattr(
            self, f"__{action_handler.__name__}", ActionServer(self, idl, namespace, cb)
        )