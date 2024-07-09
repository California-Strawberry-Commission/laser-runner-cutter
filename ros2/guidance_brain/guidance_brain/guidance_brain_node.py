import furrow_perceiver.furrow_perceiver_node as fpn
import amiga_control.amiga_control_node as acn
from guidance_brain_interfaces.srv import GetState
from guidance_brain_interfaces.msg import State
from std_srvs.srv import SetBool
from common_interfaces.msg import Vector2

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
    perceiver = import_node(fpn)
    amiga = import_node(acn)

    # State Vars
    active = False

    # Emits state.
    state = topic("~/state", State)
    
    async def emit_state(self):
        await self.state(active=self.active)

    @subscribe(amiga.amiga_available)
    async def amiga_available(self, data):
        print(data)

    @timer(0.1, False)
    async def s(self):
        print("RUN SET TWIST")
        if not self.active:
            await self.amiga.set_twist(twist=Vector2(x=0., y=0.))
            
    @subscribe(perceiver.tracker_result_topic)
    async def on_ft_result(self, linear_deviation, heading, is_valid):
        # print("FTRES", linear_deviation, heading)
        pass

    @service("~/get_state", GetState)
    async def get_state(self):
        return result(state=State(active=self.active))
    
    @service("~/set_active", SetBool)
    async def set_active(self, data: bool):
        self.active = data
        print("SET ACTIVE CALLED")

        await self.emit_state()
        return result(success=True, message="")
        

    
# Boilerplate below here.
def main():
    serve_nodes(GuidanceBrainNode())


if __name__ == "__main__":
    main()
