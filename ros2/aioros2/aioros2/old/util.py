import traceback


# Decorate sync callbacks to catch errors into the specified print function.
def catch(log_fn, return_val=None):
    def _catch(fn):
        def _safe_exec(*args, **kwargs):
            try:
                return fn(*args, **kwargs)

            except Exception:
                log_fn(traceback.format_exc())
                return return_val

        return _safe_exec

    return _catch


def idl_to_kwargs(req):
    msg_keys = req.get_fields_and_field_types().keys()
    return {k: getattr(req, k) for k in msg_keys}
