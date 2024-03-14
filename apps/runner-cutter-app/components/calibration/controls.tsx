"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";
import useCameraNode from "@/lib/useCameraNode";
import useLaserNode, { LASER_STATES } from "@/lib/useLaserNode";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const { ros } = useROS();
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
    setExposure,
  } = useCameraNode(cameraNodeName);
  const { laserState, addPoint, clearPoints, play, stop, setColor } =
    useLaserNode(laserNodeName);
  const { controlState, calibrate, addCalibrationPoint } =
    useControlNode(controlNodeName);

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  const onImageClick = (event: any) => {
    // TODO: don't do anything when calibration is happening
    const boundingRect = event.target.getBoundingClientRect();
    const x = Math.round(event.clientX - boundingRect.left);
    const y = Math.round(event.clientY - boundingRect.top);
    addCalibrationPoint(x, y);
  };

  // TODO: disable all buttons when calibration is happening
  let laserButton = null;
  if (laserState === 1) {
    laserButton = (
      <Button
        disabled={!rosConnected}
        onClick={() => {
          setColor(1.0, 0.0, 0.0);
          setExposure(0.001);
          play();
        }}
      >
        Start Laser
      </Button>
    );
  } else if (laserState === 2) {
    laserButton = (
      <Button
        disabled={!rosConnected}
        onClick={() => {
          stop();
          setExposure(-1.0);
        }}
      >
        Stop Laser
      </Button>
    );
  } else {
    laserButton = <Button disabled={true}>Laser Disconnected</Button>;
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Camera (${cameraNodeName}): connected=${connected}, laserDetectionEnabled=${laserDetectionEnabled}, runnerDetectionEnabled=${runnerDetectionEnabled}, recordingVideo=${recordingVideo}`}</p>
        <p className="text-center">{`Laser (${laserNodeName}): ${LASER_STATES[laserState]}`}</p>
        <p className="text-center">{`Control (${controlNodeName}): ${controlState}`}</p>
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
        {laserButton}
        <Button disabled={!rosConnected} onClick={() => clearPoints()}>
          Clear Points
        </Button>
      </div>
      {frameSrc && (
        <img src={frameSrc} alt="Camera Color Frame" onClick={onImageClick} />
      )}
    </div>
  );
}
