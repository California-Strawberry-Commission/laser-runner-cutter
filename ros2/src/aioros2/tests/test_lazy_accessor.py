from types import SimpleNamespace

import pytest

from aioros2.lazy_accessor import LazyAccessor


def test_resolve():
    accessor = LazyAccessor().a.b.c
    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=5)))
    accessor.set_target_obj(target_obj)
    assert accessor.resolve() == 5

    target_obj = {"a": {"b": {"c": 5}}}
    accessor.set_target_obj(target_obj)
    assert accessor.resolve() == 5

    accessor = LazyAccessor(target_obj=target_obj).a.b.c
    assert accessor.resolve() == 5


def test_resolve_depth():
    accessor = LazyAccessor().a.b.c
    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=5)))
    accessor.set_target_obj(target_obj)
    assert accessor.resolve(depth=1) is target_obj.a
    assert accessor.resolve(depth=-1) is target_obj.a.b
    assert accessor.resolve(depth=0) is target_obj


def test_resolve_no_target():
    accessor = LazyAccessor()
    with pytest.raises(ValueError):
        accessor.resolve()


def test_resolve_bad_path():
    accessor = LazyAccessor().a.b.c
    target_obj = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(z=5)))
    accessor.set_target_obj(target_obj)
    with pytest.raises(AttributeError):
        accessor.resolve()

    target_obj = SimpleNamespace(a=SimpleNamespace(b=5))
    accessor.set_target_obj(target_obj)
    with pytest.raises(AttributeError):
        accessor.resolve()

    target_obj = {"a": {"b": {"z": 5}}}
    accessor.set_target_obj(target_obj)
    with pytest.raises(KeyError):
        accessor.resolve()


def test_root_accessor():
    root_accessor = LazyAccessor()
    accessor = root_accessor.a.b.c
    assert accessor.root_accessor is root_accessor
