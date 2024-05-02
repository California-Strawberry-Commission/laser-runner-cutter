from ._decorators import RosDefinition


class RosTimer(RosDefinition):
    def __init__(self, interval, skip_until_complete, fn) -> None:
        self.server_handler = fn
        self.interval = interval
        self.skip_until_complete = skip_until_complete


def timer(interval, skip_until_complete=True):
    def _timer(fn):
        return RosTimer(interval, skip_until_complete, fn)

    return _timer
