import inspect

from varname import argname


class Deferrable:
    """Magic class. When called with a variable from a function, it allows
    whatever expression was used to generate that variable's value to be reevaluated
    by calling resolve(). Useful when a value included in that expression is expected
    to change.

    IE:

    ```
    def create_deferrable(var):
        def inner(z):
            # Note: frame depth must be manually set. frame = 1 corresponds to the current (local)
            # scope.
            return Deferrable(z, frame=3)

        return inner(var)

    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=1)))
    deferrable = create_deferrable(target_obj.a.b.c)
    target_obj.a.b.c = 2
    deferrable.resolve() # -> 2
    ```
    """

    def __init__(self, var, frame: int = 1):
        var_name = argname("var", vars_only=False)
        for i in range(frame - 1):
            var_name = argname(var_name, frame=i + 2, vars_only=False)
        self._aname = var_name
        stack_frame = inspect.stack()[frame].frame
        self._globals = stack_frame.f_globals
        self._locals = stack_frame.f_locals

    def resolve(self):
        return eval(self._aname, self._globals, self._locals)
