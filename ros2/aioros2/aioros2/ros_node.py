from .util import to_snake

class RosNode:
    def __init__(self):
        self.node_name = to_snake(self.__class__.__name__)
    
    def _validate_definition(self):
        pass
