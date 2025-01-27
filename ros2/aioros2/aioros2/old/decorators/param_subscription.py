import inspect
from typing import List

from aioros2.decorators.params import RosParamReference


class RosParamSubscription:
    def __init__(self, func, param_references):
        self.references: List[RosParamReference] = param_references
        self.handler = func


def subscribe_param(*param_references):
    """
    A function decorator for a function that will be run whenever any of the specified params change.

    Args:
        *param_references: List of param fields for which changes will trigger the function call.
    Raises:
        TypeError: If the decorated object is not a function.
    """

    def _subscribe_param(func):
        if not inspect.isfunction(func):
            raise TypeError("This decorator can only be applied to functions.")

        return RosParamSubscription(func, param_references)

    return _subscribe_param
