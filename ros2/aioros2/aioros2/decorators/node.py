from ..util import to_snake

class RosNode:        
    def _validate_definition(self):
        pass

def node(executable: str) -> RosNode:
    def _node(cls):
        class _RosNode(RosNode):
            _aioros2_executable = executable

        return type(cls.__name__, (_RosNode, cls), {})
    return _node
