"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";
import useCameraNode from "@/lib/useCameraNode";
import useLaserNode, { LASER_STATES } from "@/lib/useLaserNode";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const { ros, callService } = useROS();
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select node names
  const [cameraNodeName, setCameraNodeName] = useState<string>("/camera0");
  const [laserNodeName, setLaserNodeName] = useState<string>("/laser0");
  const [controlNodeName, setControlNodeName] = useState<string>("/control0");
  const {
    connected,
    laserDetectionEnabled,
    runnerDetectionEnabled,
    recordingVideo,
    frameSrc,
  } = useCameraNode(cameraNodeName);
  const { laserState } = useLaserNode(laserNodeName);
  const { calibrate } = useControlNode(controlNodeName);

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Camera (${cameraNodeName}): connected=${connected}, laserDetectionEnabled=${laserDetectionEnabled}, runnerDetectionEnabled=${runnerDetectionEnabled}, recordingVideo=${recordingVideo}`}</p>
        <p className="text-center">{`Laser (${laserNodeName}): ${LASER_STATES[laserState]}`}</p>
      </div>
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={!rosConnected}
          onClick={() => {
            calibrate();
          }}
        >
          Start Calibration
        </Button>
      </div>
      {frameSrc && <img src={frameSrc} alt="Camera Color Frame" />}
    </div>
  );
}
