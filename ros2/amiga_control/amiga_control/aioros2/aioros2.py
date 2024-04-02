import asyncio
import dataclasses
import re
from typing import Optional
import rclpy
import rclpy.node
from rclpy.action import ActionServer
import atexit
import inspect
from .server_driver import ServerDriver

# https://stackoverflow.com/questions/338101/python-function-attributes-uses-and-abuses
# https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process


"""
Done:
Action Server
Service Server

Service Client


Todo:
Action Client
Topic Publisher
Topic Subscriber
Server background tasks
Namespace linking
Server param change subscriptions
Client param sets
Server param side effects
Server timer tasks

"""
async def _ros_spin_nodes(nodes):
    print("Ros starting up...")

    # TODO: Tune thread counts. Might be limitations - not sure what ROS behavior when running out.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=10)

    for node in nodes:
        executor.add_node(node)

    print("Ros event loop running!")
    while rclpy.ok():
        executor.spin_once(timeout_sec=0)
        await asyncio.sleep(1e-4)


def serve_nodes(*nodes):
    rclpy.init()

    servers = [ServerDriver(n) for n in nodes]
    tasks = [task for s in servers for task in s.tasks()]

    tasks = asyncio.wait(tasks + [_ros_spin_nodes(servers)])
    asyncio.get_event_loop().run_until_complete(tasks)