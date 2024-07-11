import furrow_perceiver.furrow_perceiver_node as fpn
import amiga_control.amiga_control_node as acn

from std_srvs.srv import SetBool
from common_interfaces.msg import Vector2, PID

from guidance_brain_interfaces.msg import State
from guidance_brain_interfaces.srv import SetPID 


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
    latched_topic
)

# Executable to call to launch this node (defined in `setup.py`)
@node("guidance_brain")
class GuidanceBrainNode:
    perceiver = import_node(fpn)
    amiga = import_node(acn)

    # State Vars
    state = State(
        guidance_active = False,
        follower_pid = PID(),
    )

    # Emits state.
    state_topic = latched_topic("~/state", State)
    
    async def emit_state(self):
        await self.state_topic(self.state)

    @timer(0.1, False)
    async def s(self):        
        if not self.amiga.amiga_available.value:
            return
        
        if not self.state.guidance_active:
            await self.amiga.set_twist(twist=Vector2(x=0., y=0.))
            
    @subscribe(perceiver.tracker_result_topic)
    async def on_ft_result(self, linear_deviation, heading, is_valid):
        # print("FTRES", linear_deviation, heading)
        pass
    
    @service("~/set_active", SetBool)
    async def set_active(self, data: bool):
        self.state.guidance_active = data
        print("SET ACTIVE CALLED")

        await self.emit_state()
        
        return result()
        
    @service("~/set_follower_pid", SetPID)
    async def set_pid(self, pid: PID):
        print("SET PID CALLED")
        self.state.follower_pid = pid

        await self.emit_state()
        
        return result()
    
# Boilerplate below here.
def main():
    serve_nodes(GuidanceBrainNode())


if __name__ == "__main__":
    main()
