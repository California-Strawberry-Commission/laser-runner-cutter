from ._decorators import RosDefinition

class RosTimer(RosDefinition):
    def __init__(self, interval, fn) -> None:
        self.server_handler = fn
        self.interval = interval

def timer(interval):
    def _timer(fn):
        return RosTimer(interval, fn)
    return _timer

    