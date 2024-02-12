"""Template module with math functions."""


def add(a: int, b: int) -> int:
    """Template function to add to integer values."""
    assert isinstance(a, int), f"not an integer. Got: {type(a)}"
    assert type(a) == type(b), f"Type of 'b' must be equal. Got: {type(b)}"
    return a + b
