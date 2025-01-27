import inspect

from varname import argname


class Deferrable:
    """Magic class. When called with a variable from a function, it allows
    whatever expression was used to generate that variable's value to be reevaluated
    by calling resolve(). Useful when a value included in that expression is expected
    to change.

    IE:

    ```
    def f(var)
        return Deferrable(var)

    x = 5
    d = f(x + 5)
    d.resolve() # -> 10

    x = 10

    # Returns as though `x=10` in `x + 5`
    d.resolve() # -> 15
    ```
    """

    def __init__(self, var):
        v = argname("var")
        self._aname = argname(v, frame=2, vars_only=False)
        self._globals = inspect.stack()[2].frame.f_globals

    def resolve(self):
        return eval(self._aname, self._globals)
