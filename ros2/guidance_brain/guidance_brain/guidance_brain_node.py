import furrow_perceiver.furrow_perceiver_node as fpn
from guidance_brain_interfaces.srv import GetState
from guidance_brain_interfaces.msg import State

from aioros2 import (
    timer,
    service,
    action,
    serve_nodes,
    result,
    feedback,
    subscribe,
    topic,
    import_node,
    params,
    node,
    subscribe_param,
    param,
    start,
)

# Executable to call to launch this node (defined in `setup.py`)
@node("guidance_brain")
class GuidanceBrainNode:
    state = topic("~/state", State)

    perceiver = import_node(fpn)
    
    @start
    async def s(self):
        self.log("STARTING BRAIN")
    
    @subscribe(perceiver.track_result)
    async def on_ft_result(self, linear_deviation, heading):
        print("FTRES", linear_deviation, heading)

    @service("get_state", GetState)
    async def get_state(self):
        return result(state=State(fps=0, camera_connected=False))
    
# Boilerplate below here.
def main():
    serve_nodes(GuidanceBrainNode())


if __name__ == "__main__":
    main()
