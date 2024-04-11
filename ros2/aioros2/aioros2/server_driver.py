import asyncio
import dataclasses
import rclpy
from rclpy.action import ActionServer
from .async_driver import AsyncDriver
from .result import Result
from .feedback import Feedback

from .decorators.self import Self
from .decorators import RosDefinition
from .decorators.service import RosService
from .decorators.topic import RosTopic
from .decorators.subscribe import RosSubscription
from .decorators.import_node import RosImport
from .decorators.action import RosAction
from .decorators.timer import RosTimer
from .decorators.param import RosParamHandler
from .decorators import idl_to_kwargs

class ServerDriver(AsyncDriver):
    def __init__(self, async_node):
        super().__init__(async_node)
        
        # Get ros-asyncio decorated functions to bind.
        for handler in self._get_ros_definitions():
            print(handler)

        self._attach()


    def _process_import(self, attr, imp: RosImport):
        from .client_driver import ClientDriver

        self.log.info("Resolving SERVER import")

        return ClientDriver(imp.resolve())


    def _attach_service(self, attr, srv_def: RosService):
        self.log.info(f"Attach service @ >{srv_def.namespace}<")

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
        self.log.info(f"Attach action >{attr}<")

        def cb(goal):
            print(f"ACTION HANDLER @ {d.namespace} START", goal)

            kwargs = idl_to_kwargs(goal.request)
            gen = d.sever_handler(**kwargs)

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
            self, f"__{d.sever_handler.__name__}", ActionServer(self, d.idl, d.namespace, cb)
        )
    
    def _attach_subscriber(self, attr, sub: RosSubscription): 
        topic = sub.get_topic(self)

        self.log.info(f"Attach subscription to namespace >{topic.namespace}<")

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

        def _dispatch_pub(*args, **kwargs):
            msg = topic_def.idl(*args, **kwargs)
            pub.publish(msg)
            
        return _dispatch_pub