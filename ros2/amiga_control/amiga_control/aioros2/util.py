import re


def to_camel_case(snake_str):
    # https://stackoverflow.com/questions/19053707/converting-snake-case-to-lower-camel-case-lowercamelcase
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_snake(camel_str):
    camel_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()

