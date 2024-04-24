from typing import List
from ._decorators import RosDefinition
from .params import RosParamReference

class RosParamSubscription(RosDefinition):
    def __init__(self, handler, param_references):
        self.references: List[RosParamReference] = param_references
        self.handler = handler


def subscribe_param(*args):
    def _subscribe_param(fn):
        return RosParamSubscription(fn, args)
    return _subscribe_param
