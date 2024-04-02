import inspect
from ._decorators import decorate_handler, ros_type_e

def on(topic):
    def _on(fn):
        return decorate_handler(fn, ros_type_e.TOPIC_SUBSCRIBER)

    return _on
