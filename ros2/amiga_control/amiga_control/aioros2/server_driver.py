import asyncio
import dataclasses
import rclpy
from rclpy.action import ActionServer
from .async_driver import AsyncDriver
from .decorators._decorators import ros_type_e
from .result import Result
from .feedback import Feedback

class ServerDriver(AsyncDriver):
    def __init__(self, async_node):
        super().__init__(async_node)
        
        # Get ros-asyncio decorated functions to bind.
        for handler in self._get_decorated():
            print(handler)

    def tasks(self):
        tasks = []
        
        attachers = {
            ros_type_e.ACTION: self._attach_action,
            ros_type_e.SERVICE: self._attach_service
        }

        for handler in self._get_decorated():
            try:
                attachers[handler._ros_type](handler)
            except KeyError:
                self.log.warn(f"Could not initialize handler >{handler.__name__}< because type >{handler._ros_type}< is unknown.")

        return tasks

    def _attach_service(self, service_handler):
        self.log.info(f"Attach service >{service_handler.__name__}<")

        handler_name = service_handler.__name__
        idl = service_handler._ros_idl
        namespace = service_handler._ros_namespace

        def cb(req, res):
            print(f"SERVICE HANDLER {namespace} START", req)

            msg_keys = req.get_fields_and_field_types().keys()

            kwargs = {k: getattr(req, k) for k in msg_keys}

            result = asyncio.run_coroutine_threadsafe(
                service_handler(**kwargs), loop=self._loop
            ).result()

            # Prevent ROS hang if return value is invalid
            if not isinstance(result, Result):
                self.log.warn(
                    f"Service handler {handler_name} did not return `result(...)`. "
                    "Expected Result, got {result}"
                )
                result = res
            else:
                result = idl.Response(*result.args, **result.kwargs)

            print(f"SERVICE HANDLER {namespace} FINISH", result)

            return result

        self.create_service(idl, namespace, cb)

    def _attach_action(self, action_handler):
        self.log.info(f"Attach action >{action_handler.__name__}<")
        handler_name = action_handler.__name__
        idl = action_handler._ros_idl
        namespace = action_handler._ros_namespace

        def cb(goal):
            print(f"ACTION HANDLER {namespace} START", goal)
            req = goal.request
            goal_keys = req.get_fields_and_field_types().keys()

            kwargs = {k: getattr(req, k) for k in goal_keys}

            gen = action_handler(**kwargs)

            result: Feedback | Result = None
            while True:
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        gen.__anext__(),
                        loop=self._loop,
                    ).result()

                    print(f"ACTION HANDLER {namespace} FEEDBACK", result)

                    if isinstance(result, Feedback):
                        fb = idl.Feedback(*result.args, **result.kwargs)
                        goal.publish_feedback(fb)
                        
                    elif isinstance(result, Result):
                        goal.succeed()
                        break
                    
                    else:
                        self.log.error("YIELDED NON-FEEDBACK, NON-RESULT VALUE")
                        
                except StopAsyncIteration:
                    self.log.warn(f"Action >{handler_name}< returned before yielding `result`")
                    break


            if not isinstance(result, Result):
                self.log.warn(
                    f"Action >{handler_name}< did not yield `result(...)`. "
                    f"Expected >{idl.Result}<, got >{result}<. Aborting action."
                )
                result = idl.Result()
            else:
                result = idl.Result(*result.args, **result.kwargs)

            print(f"ACTION HANDLER {namespace} FINISH", result)
            return result

        setattr(
            self, f"__{action_handler.__name__}", ActionServer(self, idl, namespace, cb)
        )