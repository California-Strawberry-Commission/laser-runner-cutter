import inspect


class RosTimer:
    def __init__(self, interval_secs, allow_concurrent_execution, func) -> None:
        self.func = func
        self.interval_secs = interval_secs
        self.allow_concurrent_execution = allow_concurrent_execution


def timer(interval_secs: float, allow_concurrent_execution: bool = True):
    """
    A function decorator for functions that will run at regular intervals.

    Args:
        interval_secs (float): Interval between function calls, in seconds.
        allow_concurrent_execution (bool): If false, the next call will occur concurrently even if the previous call has not completed yet. If true, the next call will be skipped if the previous call has not completed yet.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _timer(func) -> RosTimer:
        if not inspect.isfunction(func):
            raise TypeError("This decorator can only be applied to functions.")

        return RosTimer(interval_secs, allow_concurrent_execution, func)

    return _timer
