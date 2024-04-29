from collections import namedtuple

Result = namedtuple("Respond", ["args", "kwargs"])

def result(*args, **kwargs):
    return Result(args, kwargs)