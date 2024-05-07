from ._decorators import RosDefinition


class RosTimer(RosDefinition):
    def __init__(self, interval, allow_concurrent_execution, fn) -> None:
        self.server_handler = fn
        self.interval = interval
        self.allow_concurrent_execution = allow_concurrent_execution


def timer(interval, allow_concurrent_execution=True):
    def _timer(fn):
        return RosTimer(interval, allow_concurrent_execution, fn)

    return _timer
