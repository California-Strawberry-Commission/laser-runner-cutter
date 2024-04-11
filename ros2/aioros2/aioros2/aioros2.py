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
Action Client

Topic Publisher
Topic Subscriber

Todo:

Namespace linking
Server background tasks
Server param change subscriptions
Client param sets
Server param side effects
Server timer tasks
"""
async def _ros_spin_nodes(nodes, num_threads):
    print("Ros starting up...")

    # TODO: Tune thread counts. Might be limitations - not sure what ROS behavior when running out.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)

    servers = [ServerDriver(n) for n in nodes]

    for node in servers:
        executor.add_node(node)

    print("Ros event loop running!")
    while rclpy.ok():
        executor.spin_once(timeout_sec=0)
        await asyncio.sleep(1e-4)


def serve_nodes(*nodes, threads=10):
    rclpy.init()

    asyncio.run(_ros_spin_nodes(nodes, threads))