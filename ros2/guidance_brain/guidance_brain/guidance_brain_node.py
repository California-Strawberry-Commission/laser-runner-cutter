from dataclasses import dataclass
import time
import furrow_perceiver.furrow_perceiver_node as fpn
import amiga_control.amiga_control_node as acn

from std_srvs.srv import SetBool, Trigger
from common_interfaces.msg import Vector2, PID
from common_interfaces.srv import SetFloat32

from guidance_brain_interfaces.msg import State
from guidance_brain_interfaces.srv import SetPID 
from furrow_perceiver_interfaces.msg import PositionResult
from enum import IntEnum
import asyncio

from aioros2 import (
    timer,
    service,
    serve_nodes,
    subscribe,
    topic,
    import_node,
    node,
    QOS_LATCHED,
)

class GoDirection(IntEnum):
    FORWARD=0
    BACKWARD=1

# Executable to call to launch this node (defined in `setup.py`)
@node("guidance_brain")
class GuidanceBrainNode:
    perceiver_forward = import_node(fpn, "furrow0")
    # perceiver_backward = import_node(fpn, "furrow1")

    amiga = import_node(acn)
        
    # State Vars
    state = State(
        guidance_active = False,
        amiga_connected = False,
        
        speed = 20., # Default - 10ft/min
        follower_pid = PID(p=50.),
        
        # Keeps selected (forward/backward) perciever result
        perceiver_valid = False,
        error = 0.,
        
        command = 0.,
        
        go_direction = GoDirection.FORWARD,
        go_last_valid_time = 0.,
    )
    
    # Emits state.
    state_topic = topic("~/state", State, QOS_LATCHED)

    async def emit_state(self):
        await self.state_topic(self.state)

    
    @timer(0.05, False)
    async def s(self):      
        await self.emit_state()

        # Feed timeout
        if self.state.perceiver_valid:
            self.state.go_last_valid_time = time.time()
                
        # 1 second has passed since furrow perciever was valid - kill following
        if self.state.guidance_active and (time.time() - self.state.go_last_valid_time > 1):
            self.state.guidance_active = False
        
        # Short circuit if perciever isn't valid.
        if not (self.state.guidance_active and self.state.perceiver_valid):
            self.state.command = 0.
            await self.amiga.set_twist(twist=Vector2(x=0., y=0.))
            return

        # Run PID
        self.state.command = self.state.follower_pid.p * self.state.error / 5000.
        speed_ms = self.state.speed * 0.00508
        
        if self.state.go_direction == GoDirection.BACKWARD:
            speed_ms = -speed_ms
            
        await self.amiga.set_twist(twist=Vector2(
            x=self.state.command,
            y=speed_ms
        ))

    @subscribe(amiga.amiga_available)
    async def on_amiga_available(self, data):
        self.state.amiga_connected = data
    
    @subscribe("/furrow1/tracker_result", PositionResult)
    async def on_fp_back_result(self, linear_deviation, heading, is_valid):
        if self.state.go_direction == GoDirection.BACKWARD:
            self.state.perceiver_valid = is_valid
            self.state.error = linear_deviation
            
    @subscribe(perceiver_forward.tracker_result_topic)
    async def on_fp_forw_result(self, linear_deviation, heading, is_valid):
        if self.state.go_direction != GoDirection.BACKWARD:
            self.state.perceiver_valid = is_valid
            self.state.error = linear_deviation
        
    @service("~/set_p", SetFloat32)
    async def set_p(self, data: float):
        self.state.follower_pid.p = data
        return {"success": True}

    @service("~/set_i", SetFloat32)
    async def set_i(self, data: float):
        self.state.follower_pid.i = data
        return {"success": True}

    @service("~/set_d", SetFloat32)
    async def set_d(self, data: float):
        self.state.follower_pid.d = data
        return {"success": True}

    @service("~/set_speed", SetFloat32)
    async def set_speed(self, data: float):
        self.state.speed = data
        return {"success": True}
    
    @service("~/go_forward", Trigger)
    async def go_forward(self):
        self.state.go_last_valid_time = time.time()
        self.state.guidance_active = True
        self.state.go_direction = GoDirection.FORWARD  
     
        return {"success": True}
    
    @service("~/go_backward", Trigger)
    async def go_backward(self):
        self.state.go_last_valid_time = time.time()
        self.state.guidance_active = True
        self.state.go_direction = GoDirection.BACKWARD  
        return {"success": True}
    
    @service("~/stop", Trigger)
    async def stop(self):
        self.state.guidance_active = False
        return {"success": True}
    
# Boilerplate below here.
def main():
    serve_nodes(GuidanceBrainNode())


if __name__ == "__main__":
    main()


# 30 ft/min
# 50 pid

# FORWARD
# -12 offset
