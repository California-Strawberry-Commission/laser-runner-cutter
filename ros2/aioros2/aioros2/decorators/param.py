from ._decorators import RosDefinition

class RosParamHandler(RosDefinition):
    pass
def param(dataclass_param):
    def _param(fn):
        return fn

    return _param