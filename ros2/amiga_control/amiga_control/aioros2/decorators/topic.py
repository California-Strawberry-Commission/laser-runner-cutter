from typing import Any
from ._decorators import decorate_handler, ros_type_e

def topic(namespace: str, idl: Any, qos: int):
    def _dispatch(self, *args, **kwargs):
        msg = idl(*args, **kwargs)
        self.publisher.publish(msg)
        
    return decorate_handler(_dispatch, ros_type_e.TOPIC, namespace=namespace, idl=idl, qos=qos)