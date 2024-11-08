from typing import List, Optional

from rclpy.parameter import Parameter


class RosNode:
    pass


def node(executable: str) -> RosNode:
    def _node(cls):
        def __init__(
            self,
            name: Optional[str] = None,
            namespace: Optional[str] = None,
            parameter_overrides: Optional[List[Parameter]] = None,
            *args,
            **kwargs
        ):
            super(cls, self).__init__(*args, **kwargs)
            self._aioros2_name = name
            self._aioros2_namespace = namespace
            self._aioros2_parameter_overrides = parameter_overrides

        # Create a new class that inherits from both RosNode and the decorated class, with
        # a custom __init__ and _aioros2_executable set
        return type(
            cls.__name__,
            (RosNode, cls),
            {"__init__": __init__, "_aioros2_executable": executable},
        )

    return _node
