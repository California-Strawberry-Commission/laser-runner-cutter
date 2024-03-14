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
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  const onImageLoad = (event: any) => {
    const { naturalWidth: width, naturalHeight: height } = event.target;
    setImageSize({ width, height });
  };

  const onImageClick = (event: any) => {
    if (controlState !== "idle") {
      return;
    }

    const boundingRect = event.target.getBoundingClientRect();
    const x = Math.round(event.clientX - boundingRect.left);
    const y = Math.round(event.clientY - boundingRect.top);
    // Scale x, y from rendered size to actual image size
    const scaledX = (imageSize.width / boundingRect.width) * x;
    const scaledY = (imageSize.height / boundingRect.height) * y;
    addCalibrationPoint(scaledX, scaledY);
  };

  let laserButton = null;
  if (laserState === 1) {
    laserButton = (
      <Button
        disabled={!rosConnected || controlState !== "idle"}
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
        disabled={!rosConnected || controlState !== "idle"}
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
          disabled={!rosConnected || controlState !== "idle"}
          onClick={() => {
            calibrate();
          }}
        >
          Start Calibration
        </Button>
        {laserButton}
        <Button
          disabled={!rosConnected || controlState !== "idle"}
          onClick={() => clearPoints()}
        >
          Clear Points
        </Button>
      </div>
      {frameSrc && (
        <img
          src={frameSrc}
          alt="Camera Color Frame"
          onLoad={onImageLoad}
          onClick={onImageClick}
        />
      )}
    </div>
  );
}
