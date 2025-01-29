# aioros2

The point of this library is to make working with ROS2/`rclpy` bearable (and possibly even enjoyable). Its syntax is heavily inspired by `python-socketio`.

## Features

- Massive boilerplate reduction
- First-class asyncio compatibility
- Generators instead of callbacks
- Transparent clients (use the same class as both client and server)

### TODO

- [x] Service Server
- [x] Service Client
- [x] Topic Publisher
- [x] Topic Subscriber
- [x] Server timer tasks
- [x] Launch files
- [x] Server background tasks
- [x] Parameters
- [ ] Parameter subscriptions
- [ ] Action Server
- [ ] Action Client
- [ ] Comprehensive error handling
- [x] Circular imports
- [ ] Launching multiple nodes in a single process

### Known Limitations

- Param dataclasses must be flat
- Non-async handlers are not currently supported
- Improve validation, error handling, and error messaging

## Motivation

Here is a comparison between the [example ROS 2 simple publisher and subscriber](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html) and a fully-featured equivalent using `aioros2`:

<table>
<tr>
<th>rclpy</th>
<th>aioros2</th>
</tr>
<tr>
<td>

```python
# publisher.py
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

</td>
<td>

```python
# publisher.py
import aioros2

from std_msgs.msg import String


my_topic = aioros2.topic("~/topic", String, 10)


@aioros2.start
async def start(node):
    node.i = 0


@aioros2.timer(0.5)
async def timer(node):
    msg = String()
    msg.data = 'Hello World: %d' % self.i
    my_topic.publish(msg)
    node.get_logger().info('Publishing: "%s"' % msg.data)
    node.i += 1


def main():
    aioros2.run()


if __name__ == "__main__":
    main()

```

</td>
</tr>
<tr>
<td>

```python
# subscriber.py
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

</td>
<td>

```python
# subscriber.py
import aioros2

from std_msgs.msg import String
import .publisher as publisher


publisher_ref = aioros2.use(publisher)


# We can directly reference the publisher node's topic. The fully qualified
# topic name will be resolved automatically.
@aioros2.subscribe(publisher_ref.my_topic)
async def on_topic_msg(node, data):
    node.get_logger().info('I heard: "%s"' % data)


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
```

</td>

</table>

## Installation

This will install `aioros2` in dev mode in the current environment

```
$ cd aioros2
$ pip install -e . --config-settings editable_mode=strict
```
