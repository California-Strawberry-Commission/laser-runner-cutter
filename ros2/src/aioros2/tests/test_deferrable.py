from types import SimpleNamespace

import pytest

from aioros2.deferrable import Deferrable


def test_resolve():
    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=1)))
    deferrable = Deferrable(target_obj.a.b.c)
    target_obj.a.b.c = 2
    assert deferrable.resolve() is target_obj.a.b.c

    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=1)))
    deferrable = Deferrable(target_obj.a.b.c + 5)
    target_obj.a.b.c = 2
    assert deferrable.resolve() is target_obj.a.b.c + 5


def test_resolve_nested():
    def create_deferrable(var):
        def inner(z):
            return Deferrable(z, frame=3)

        return inner(var)

    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=1)))
    deferrable = create_deferrable(target_obj.a.b.c)
    target_obj.a.b.c = 2
    assert deferrable.resolve() is target_obj.a.b.c

    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=1)))
    deferrable = create_deferrable(target_obj.a.b.c + 5)
    target_obj.a.b.c = 2
    assert deferrable.resolve() is target_obj.a.b.c + 5
