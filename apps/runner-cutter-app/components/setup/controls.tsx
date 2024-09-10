"use client";

import FramePreview from "@/components/camera/frame-preview";
import NodeCards from "@/components/nodes/node-cards";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import useROS from "@/lib/ros/useROS";
import useLifecycleManagerNode from "@/lib/useLifecycleManagerNode";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode, {
  DeviceState as LaserDeviceState,
} from "@/lib/useLaserNode";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";

export default function Controls() {
  const { connected: rosConnected } = useROS();

  const lifecycleManagerNode = useLifecycleManagerNode("/lifecycle_manager");
  const cameraNode = useCameraNode("/camera0");
  const laserNode = useLaserNode("/laser0");
  const controlNode = useControlNode("/control0");

  const nodeInfos = useMemo(() => {
    const rosbridgeNodeInfo = {
      name: "Rosbridge",
      connected: rosConnected,
    };
    return [
      rosbridgeNodeInfo,
      lifecycleManagerNode,
      cameraNode,
      laserNode,
      controlNode,
    ];
  }, [rosConnected, lifecycleManagerNode, cameraNode, laserNode, controlNode]);

  let cameraButton = null;
  const enableCameraButton = cameraNode.connected;
  if (cameraNode.state.deviceState === CameraDeviceState.Disconnected) {
    cameraButton = (
      <Button
        disabled={!enableCameraButton}
        onClick={() => cameraNode.startDevice()}
      >
        Connect Camera
      </Button>
    );
  } else if (cameraNode.state.deviceState === CameraDeviceState.Connecting) {
    cameraButton = (
      <Button disabled>
        <Loader2 className="h-4 w-4 animate-spin" />
      </Button>
    );
  } else {
    cameraButton = (
      <Button
        disabled={!enableCameraButton}
        variant="destructive"
        onClick={() => cameraNode.closeDevice()}
      >
        Disconnect Camera
      </Button>
    );
  }

  let laserButton = null;
  const enableLaserButton = laserNode.connected;
  if (laserNode.state === LaserDeviceState.Disconnected) {
    laserButton = (
      <Button
        disabled={!enableLaserButton}
        onClick={() => laserNode.startDevice()}
      >
        Connect Laser
      </Button>
    );
  } else if (laserNode.state === LaserDeviceState.Connecting) {
    laserButton = (
      <Button disabled>
        <Loader2 className="h-4 w-4 animate-spin" />
      </Button>
    );
  } else {
    laserButton = (
      <Button
        disabled={!enableLaserButton}
        variant="destructive"
        onClick={() => laserNode.closeDevice()}
      >
        Disconnect Laser
      </Button>
    );
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <NodeCards nodeInfos={nodeInfos} />
      <div className="flex flex-row items-center gap-4">
        {cameraButton}
        {laserButton}
        <Dialog>
          <DialogTrigger asChild>
            <Button
              disabled={!lifecycleManagerNode.connected}
              variant="destructive"
            >
              Restart Service
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Are you absolutely sure?</DialogTitle>
              <DialogDescription>
                This will restart the ROS 2 nodes, and may take a few minutes
                for it to come back up. You can also choose to reboot the host
                machine, which will take longer.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <DialogClose asChild>
                <Button
                  variant="destructive"
                  onClick={() => {
                    lifecycleManagerNode.restart_service();
                  }}
                >
                  Restart Nodes
                </Button>
              </DialogClose>
              <DialogClose asChild>
                <Button
                  variant="destructive"
                  onClick={() => {
                    lifecycleManagerNode.reboot_system();
                  }}
                >
                  Reboot System
                </Button>
              </DialogClose>
              <DialogClose asChild>
                <Button>Cancel</Button>
              </DialogClose>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
      <FramePreview topicName={"/camera0/debug_frame"} />
    </div>
  );
}
