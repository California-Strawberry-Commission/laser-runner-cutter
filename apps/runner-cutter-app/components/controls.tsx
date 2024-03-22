"use client";

import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";
import useCameraNode from "@/lib/useCameraNode";
import useLaserNode, { LASER_STATES } from "@/lib/useLaserNode";
import useControlNode from "@/lib/useControlNode";
import { Button } from "@/components/ui/button";

export default function Controls() {
  const ros = useContext(ROSContext);
  const [rosConnected, setRosConnected] = useState<boolean>(
    ros.ros.isConnected
  );
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
  const { calibrated, controlState, calibrate, startRunnerCutter, stop } =
    useControlNode(controlNodeName);

  useEffect(() => {
    ros.onStateChange(() => {
      setRosConnected(ros.ros.isConnected);
    });
  }, [setRosConnected]);

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Camera (${cameraNodeName}): connected=${connected}, laserDetectionEnabled=${laserDetectionEnabled}, runnerDetectionEnabled=${runnerDetectionEnabled}, recordingVideo=${recordingVideo}`}</p>
        <p className="text-center">{`Laser (${laserNodeName}): ${LASER_STATES[laserState]}`}</p>
        <p className="text-center">{`Control (${controlNodeName}): calibrated=${calibrated}, controlState=${controlState}`}</p>
      </div>
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={!rosConnected || controlState !== "idle"}
          onClick={() => {
            calibrate();
          }}
        >
          Calibrate
        </Button>
        <Button
          disabled={!rosConnected || controlState !== "idle"}
          onClick={() => {
            startRunnerCutter();
          }}
        >
          Start Cutter
        </Button>
        <Button
          disabled={!rosConnected || controlState === "idle"}
          variant="destructive"
          onClick={() => {
            stop();
          }}
        >
          Stop
        </Button>
      </div>
      {frameSrc && <img src={frameSrc} alt="Camera Color Frame" />}
    </div>
  );
}
