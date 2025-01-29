from __future__ import annotations

from typing import Any, List, Optional


class LazyAccessor:
    def __init__(
        self,
        path: Optional[List[str]] = None,
        root_accessor: Optional[LazyAccessor] = None,
        target_obj: Optional[Any] = None,
    ):
        """
        Initialize a LazyAccessor with an optional path.

        Args:
            path (Optional[List[str]]): List of attribute keys representing the path.
            root_accessor (Optional[Any]): The root LazyAccessor instance on which the path is based on.
            target_obj (Optional[Any]): The object (or dictionary) on which to resolve the path.
        """
        self._path = path or []
        # If the root accessor is not defined and the path is empty, use self as the root accessor
        self._root_accessor = (
            self if root_accessor is None and not self._path else root_accessor
        )
        self._target_obj = target_obj

    @property
    def root_accessor(self) -> Optional[LazyAccessor]:
        """
        Get the root LazyAccessor instance on which the path is based on.

        Returns:
            Optional[LazyAccessor]: The root LazyAccessor instance on which the path is based on.
        """
        return self._root_accessor

    @property
    def path(self) -> List[str]:
        """
        Get the list of attribute keys associated with this accessor.

        Returns:
            List[str]: The list of attribute keys associated with this accessor.
        """
        return self._path

    def __getattr__(self, name: str):
        """
        Called when an attribute is accessed. Returns a new LazyAccessor
        with the updated path including the accessed attribute.
        """
        return LazyAccessor(self._path + [name], self._root_accessor, self._target_obj)

    def set_target_obj(self, target_obj: Any) -> LazyAccessor:
        """
        Set the object instance on which to resolve the path when resolve() is called.

        Args:
            target_obj (Any): The object (or dictionary) on which to resolve the path.
        Returns:
            self
        """
        self._target_obj = target_obj
        return self

    def resolve(self, depth: Optional[int] = None) -> Any:
        """
        Resolves the stored path on the provided object instance.

        Args:
            depth (Optional[int]): Depth to traverse. Positive depth goes up to that many elements. Negative depth excludes that many elements from the end. None (default) traverses the entire path. Zero depth just returns the target object.
        Returns:
            The value at the resolved path.
        Raises:
            KeyError or AttributeError: If the path cannot be resolved.
            ValueError: If the target object instance has not been set yet.
        """
        if self._target_obj is None:
            raise ValueError(
                "Target object instance on which to resolve has not been set."
            )

        if depth is None:
            effective_path = self._path
        elif depth > 0:
            effective_path = self._path[:depth]
        elif depth < 0:
            effective_path = self._path[: len(self._path) + depth]
        else:
            return self._target_obj

        value = self._target_obj
        for key in effective_path:
            if isinstance(value, dict):
                value = value[key]
            else:
                value = getattr(value, key)
        return value
