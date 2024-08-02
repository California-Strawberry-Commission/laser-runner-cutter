# aioros2
The point of this library is to make working with ROS2/`rclpy` bearable (and possibly even enjoyable). Its syntax is heavily inspired by `python-socketio`.

## Features
- Massive boilerplate reduction
- First-class asyncio compatibility.
- Generators instead of callbacks
- Transparent clients (use the same class as both client and server)

## Installation
This will install `aioros2` in dev mode in the current environment
```
$ cd aioros2
$ pip install -e . --config-settings editable_mode=strict
```

### Feature Tracker
- [x] Action Server
- [x] Service Server
- [x] Service Client
- [x] Action Client
- [x] Topic Publisher
- [x] Topic Subscriber
- [x] Server timer tasks
- [x] Namespace linking
    - Need to properly resolve ROS namespaces (including `~`) to other nodes
- [x] Remapping
    - Want to provide helpers to make linking nodes in launch files easy.
    - ` link(node1.dep_node_1, node2.dep_node_2) ` -> list of remaps to use in launch.py
- [x] Server param side effects
- [x] 2+ order import topic resolution
- [ ] Non-async handlers / better warnings?
- [x] Server background tasks
- [ ] Comprehensive error handling

## Limitations
- Param dataclasses must be flat.
- Non-async handlers are not currently supported
- Probably fragile. Next steps are improving validation, error handling, and error messaging

## Why?
Here's a comparison between the [example ROS2 action client/server](https://docs.ros.org/en/foxy/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html) and a fully-featured equivalent using `aioros2`:

<table>
<tr>
<th> aioros2 </th> <th> rclpy </th>
</tr>
<tr>
<td> 

```python
# server.py
import asyncio
from action_tutorials_interfaces.action import Fibonacci
from aioros2 import node, action, serve_nodes, result, feedback

@node()
class Fibonacci:

    @action("~/fibonacci", Fibonacci)
    async def action_fib(self, order):
        sequence = [0, 1]
        for i in range(order):
            sequence.append(sequence[-1] + sequence[-2])
            yield feedback(partial_sequence=sequence)
            await asyncio.sleep(1)

        # Last yield is result
        yield result(sequence=sequence)
        

def main():
    serve_nodes(Fibonacci())

if __name__ == "__main__":
    main()
```
```python
# client.py
import asyncio
from .server import Fibonacci
from aioros2 import ClientDriver

async def _main():
    n = ClientDriver(Fibonacci())

    print("Calling action!")
    action = n.action_fib(order=10)
    async for feedback in action:
        print("Got partial sequence:", feedback.partial_sequence)
    print("Got result: ", action.result.sequence)

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()

```

</td>
<td> 

```python
# Server.py
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from action_tutorials_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])
            self.get_logger().info('Feedback: {0}'.format(feedback_msg.partial_sequence))
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()

        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result


def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    rclpy.spin(fibonacci_action_server)


if __name__ == '__main__':
    main()
```
```python
# Client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_tutorials_interfaces.action import Fibonacci


class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.sequence))
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.partial_sequence))


def main(args=None):
    rclpy.init(args=args)

    action_client = FibonacciActionClient()

    action_client.send_goal(10)

    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
```

</td>
</table>

1/2 the lines, 10x the readability. Enough said :)