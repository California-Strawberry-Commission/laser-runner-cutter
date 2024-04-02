from ._decorators import decorate_handler, ros_type_e

def param(dataclass_param):
    def _param(fn):
        decorate_handler(fn, ros_type_e.PARAM)
        return fn

    return _param