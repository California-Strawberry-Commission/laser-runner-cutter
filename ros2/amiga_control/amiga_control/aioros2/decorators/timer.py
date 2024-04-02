from ._decorators import decorate_handler, ros_type_e

def timer(interval):
    def _timer(fn):
        decorate_handler(fn, ros_type_e.TIMER)
        return fn

    return _timer