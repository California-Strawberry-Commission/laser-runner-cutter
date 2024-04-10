import asyncio
import dataclasses
import rclpy
from rclpy.action import ActionServer
from .async_driver import AsyncDriver
from .result import Result
from .feedback import Feedback

from .decorators.self import Self
from .decorators.service import RosService
from .decorators.topic import RosTopic
from .decorators.subscribe import RosSubscription
from .decorators.import_node import RosImport

class ServerDriver(AsyncDriver):
    def __init__(self, async_node):
        super().__init__(async_node)
        
        # Get ros-asyncio decorated functions to bind.
        for handler in self._get_ros_definitions():
            print(handler)

        # Attachers create an implementation for the passed handler which is assigned
        # to that handler's name.
        attachers = {
            # ros_type_e.ACTION: self._attach_action,
            RosService: self._attach_service,
            RosSubscription: self._attach_subscriber,
            RosTopic: self._attach_publisher,
            RosImport: self._process_import,
        }

        for attr, definition in self._get_ros_definitions():
            ros_def_class = type(definition)
            try:
                setattr(self, attr, attachers[ros_def_class](definition))
            except KeyError:
                self.log.warn(f"Could not initialize handler >{attr}< because type >{ros_def_class}< is unknown.")        

    def _process_import(self, imp: RosImport):
        from .client_driver import ClientDriver

        print("PROCESS SERVER IMPORT")

        return ClientDriver(imp.resolve())

    def _attach_service(self, srv_def: RosService):
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

    def _attach_action(self, action_handler):
        self.log.info(f"Attach action >{action_handler.__name__}<")
        handler_name = action_handler.__name__
        idl = action_handler._ros_idl
        namespace = action_handler._ros_namespace

        def cb(goal):
            print(f"ACTION HANDLER {namespace} START", goal)

            kwargs = self._idl_to_kwargs(goal.request)

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
    
    def _attach_subscriber(self, sub: RosSubscription): 
        topic = sub.get_topic(self)

        self.log.info(f"Attach subscription to namespace >{topic.namespace}<")

        def cb(msg):
            print(f"SUBSCRIPTION HANDLER {topic.namespace} START", msg)

            kwargs = self._idl_to_kwargs(msg)

            asyncio.run_coroutine_threadsafe(
                sub.server_handler(self, **kwargs), loop=self._loop
            ).result()

        self.create_subscription(topic.idl, topic.namespace, cb, topic.qos)
        
    def _attach_publisher(self, topic_def: RosTopic):
        self.log.info(f"Attach publisher @ >{topic_def.namespace}<")

        pub = self.create_publisher(topic_def.idl, topic_def.namespace, topic_def.qos)

        def _dispatch_pub(*args, **kwargs):
            msg = topic_def.idl(*args, **kwargs)
            pub.publish(msg)
            
        return _dispatch_pub