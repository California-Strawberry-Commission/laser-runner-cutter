from ._decorators import RosDefinition
from rcl_interfaces.msg import ParameterDescriptor


class RosParam(RosDefinition):
    def __init__(self, default, **kwargs):
        # self._desc = ParameterDescriptor()
        pass


def param(*args, **kwargs):
    return RosParam(*args, **kwargs)
