"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Controls() {
  const { ros, callService, subscribe } = useROS();
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select camera node name
  const [cameraNodeName, setCameraNodeName] = useState<string>("/camera0");
  const [frameSrc, setFrameSrc] = useState<string>("");
  const [exposureMs, setExposureMs] = useState<number>(0.2);

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  // Subscriptions
  useEffect(() => {
    const frameSub = subscribe(
      `${cameraNodeName}/debug_frame`,
      "sensor_msgs/CompressedImage",
      (message) => {
        setFrameSrc(`data:image/jpeg;base64,${message.data}`);
      }
    );

    return () => {
      frameSub.unsubscribe();
    };
  }, [cameraNodeName]);

  const onSetExposureClick = () => {
    callService(
      `${cameraNodeName}/set_exposure`,
      "camera_control_interfaces/SetExposure",
      { exposure_ms: exposureMs }
    );
  };

  const onAutoExposureClick = () => {
    callService(
      `${cameraNodeName}/set_exposure`,
      "camera_control_interfaces/SetExposure",
      { exposure_ms: -1.0 }
    );
  };

  const onStartLaserDetectionClick = () => {
    callService(
      `${cameraNodeName}/start_laser_detection`,
      "std_srvs/Empty",
      {}
    );
  };

  const onStopLaserDetectionClick = () => {
    callService(`${cameraNodeName}/stop_laser_detection`, "std_srvs/Empty", {});
  };

  const onStartRunnerDetectionClick = () => {
    callService(
      `${cameraNodeName}/start_runner_detection`,
      "std_srvs/Empty",
      {}
    );
  };

  const onStopRunnerDetectionClick = () => {
    callService(
      `${cameraNodeName}/stop_runner_detection`,
      "std_srvs/Empty",
      {}
    );
  };

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
      </div>
      {frameSrc && <img src={frameSrc} alt="Camera Color Frame" />}
      <div className="flex flex-row items-center gap-4">
        <Label className="flex-none w-16" htmlFor="exposure">
          Exposure (ms):
        </Label>
        <Input
          className="flex-none w-20"
          type="number"
          id="exposure"
          name="exposure"
          step={0.01}
          value={exposureMs.toString()}
          onChange={(event) => {
            const value = Number(event.target.value);
            setExposureMs(isNaN(value) ? 0 : value);
          }}
        />
        <Button disabled={!rosConnected} onClick={onSetExposureClick}>
          Set Exposure
        </Button>
        <Button disabled={!rosConnected} onClick={onAutoExposureClick}>
          Auto Exposure
        </Button>
      </div>
      <div className="flex flex-row items-center gap-4">
        <Button disabled={!rosConnected} onClick={onStartLaserDetectionClick}>
          Start Laser Detection
        </Button>
        <Button disabled={!rosConnected} onClick={onStopLaserDetectionClick}>
          Stop Laser Detection
        </Button>
      </div>
      <div className="flex flex-row items-center gap-4">
        <Button disabled={!rosConnected} onClick={onStartRunnerDetectionClick}>
          Start Runner Detection
        </Button>
        <Button disabled={!rosConnected} onClick={onStopRunnerDetectionClick}>
          Stop Runner Detection
        </Button>
      </div>
    </div>
  );
}
