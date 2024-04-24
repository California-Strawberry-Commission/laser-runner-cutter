from ._decorators import RosDefinition

class RosParam(RosDefinition):
    def __init__(self, *args):
        # self._desc = ParameterDescriptor()
        pass


def param(*args, **kwargs):
    return RosParam(*args)
