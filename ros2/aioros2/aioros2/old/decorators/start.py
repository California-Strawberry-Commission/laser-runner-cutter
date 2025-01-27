import inspect


class RosStart:
    def __init__(self, func) -> None:
        self.func = func


def start(func) -> RosStart:
    """
    A function decorator for functions that will run immediately on node start.

    Raises:
        TypeError: If the decorated object is not a function.
    """

    if not inspect.isfunction(func):
        raise TypeError("This decorator can only be applied to functions.")

    return RosStart(func)
