from collections import namedtuple

Feedback = namedtuple("Feedback", ["args", "kwargs"])

def feedback(*args, **kwargs):
    return Feedback(args, kwargs)