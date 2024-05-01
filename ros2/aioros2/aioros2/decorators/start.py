from ._decorators import RosDefinition


class RosStart(RosDefinition):
    def __init__(self, fn) -> None:
        self.server_handler = fn


def start(fn):
    return RosStart(fn)
