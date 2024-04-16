"use client";

import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";
import useCameraNode from "@/lib/useCameraNode";
import useLaserNode, { LASER_STATES } from "@/lib/useLaserNode";
import useControlNode from "@/lib/useControlNode";
import { Button } from "@/components/ui/button";
import FramePreview from "@/components/camera/frame-preview";

export default function Controls() {
  const ros = useContext(ROSContext);
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select node names
  const [cameraNodeName, setCameraNodeName] = useState<string>("/camera0");
  const [laserNodeName, setLaserNodeName] = useState<string>("/laser0");
  const [controlNodeName, setControlNodeName] = useState<string>("/control0");
  const {
    nodeConnected: cameraNodeConnected,
    cameraConnected,
    laserDetectionEnabled,
    runnerDetectionEnabled,
    recordingVideo,
    frameSrc,
  } = useCameraNode(cameraNodeName);
  const { nodeConnected: laserNodeConnected, laserState } =
    useLaserNode(laserNodeName);
  const {
    nodeConnected: controlNodeConnected,
    calibrated,
    controlState,
    calibrate,
    startRunnerCutter,
    stop,
  } = useControlNode(controlNodeName);

  useEffect(() => {
    ros.onStateChange(() => {
      setRosConnected(ros.isConnected());
    });
    setRosConnected(ros.isConnected());
  }, [ros, setRosConnected]);

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Camera (${cameraNodeName}) [${
          cameraNodeConnected ? "connected" : "disconnected"
        }]: cameraConnected=${cameraConnected}, laserDetectionEnabled=${laserDetectionEnabled}, runnerDetectionEnabled=${runnerDetectionEnabled}, recordingVideo=${recordingVideo}`}</p>
        <p className="text-center">{`Laser (${laserNodeName}) [${
          laserNodeConnected ? "connected" : "disconnected"
        }]: ${LASER_STATES[laserState]}`}</p>
        <p className="text-center">{`Control (${controlNodeName}) [${
          controlNodeConnected ? "connected" : "disconnected"
        }]: calibrated=${calibrated}, controlState=${controlState}`}</p>
      </div>
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={
            !rosConnected || !controlNodeConnected || controlState !== "idle"
          }
          onClick={() => {
            calibrate();
          }}
        >
          Calibrate
        </Button>
        <Button
          disabled={
            !rosConnected || !controlNodeConnected || controlState !== "idle"
          }
          onClick={() => {
            startRunnerCutter();
          }}
        >
          Start Cutter
        </Button>
        <Button
          disabled={
            !rosConnected || !controlNodeConnected || controlState === "idle"
          }
          variant="destructive"
          onClick={() => {
            stop();
          }}
        >
          Stop
        </Button>
      </div>
      <FramePreview frameSrc={frameSrc} />
    </div>
  );
}
