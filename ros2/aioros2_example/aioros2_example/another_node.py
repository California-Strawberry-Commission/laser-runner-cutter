from dataclasses import dataclass

from std_msgs.msg import String
from std_srvs.srv import Trigger

import aioros2


@dataclass
class MyParams:
    my_message: str = "Default"


my_params = aioros2.params(MyParams)
my_topic = aioros2.topic("~/my_topic", String, aioros2.QOS_LATCHED)


@aioros2.timer(2.0)
async def timer(node):
    my_topic.publish(data=my_params.my_message)


@aioros2.service("~/my_service", Trigger)
async def my_service(node):
    print("my_service called")
    return {"success": True}


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
