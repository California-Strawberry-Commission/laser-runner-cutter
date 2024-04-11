from ._decorators import RosDefinition

class RosTimer(RosDefinition):
    pass

def timer(interval):
    def _timer(fn):
        return fn

    return _timer