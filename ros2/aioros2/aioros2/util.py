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
