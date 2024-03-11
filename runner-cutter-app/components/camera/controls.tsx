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
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const [nodeState, setNodeState] = useState({
    connected: false,
    laser_detection_enabled: false,
    runner_detection_enabled: false,
    recording_video: false,
  });
  const [frameSrc, setFrameSrc] = useState<string>("");
  const [exposureMs, setExposureMs] = useState<number>(0.2);

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  // Initial node state
  useEffect(() => {
    const getState = async () => {
      const result = await callService(
        `${nodeName}/get_state`,
        "camera_control_interfaces/GetState",
        {}
      );
      setNodeState(result.state);
    };
    getState();
  }, [nodeName]);

  // Subscriptions
  useEffect(() => {
    const stateSub = subscribe(
      `${nodeName}/state`,
      "camera_control_interfaces/State",
      (message) => {
        setNodeState(message);
      }
    );
    const frameSub = subscribe(
      `${nodeName}/debug_frame`,
      "sensor_msgs/CompressedImage",
      (message) => {
        setFrameSrc(`data:image/jpeg;base64,${message.data}`);
      }
    );

    return () => {
      stateSub.unsubscribe();
      frameSub.unsubscribe();
    };
  }, [nodeName]);

  const onSetExposureClick = () => {
    callService(
      `${nodeName}/set_exposure`,
      "camera_control_interfaces/SetExposure",
      { exposure_ms: exposureMs }
    );
  };

  const onAutoExposureClick = () => {
    callService(
      `${nodeName}/set_exposure`,
      "camera_control_interfaces/SetExposure",
      { exposure_ms: -1.0 }
    );
  };

  const onStartLaserDetectionClick = () => {
    callService(`${nodeName}/start_laser_detection`, "std_srvs/Empty", {});
  };

  const onStopLaserDetectionClick = () => {
    callService(`${nodeName}/stop_laser_detection`, "std_srvs/Empty", {});
  };

  const onStartRunnerDetectionClick = () => {
    callService(`${nodeName}/start_runner_detection`, "std_srvs/Empty", {});
  };

  const onStopRunnerDetectionClick = () => {
    callService(`${nodeName}/stop_runner_detection`, "std_srvs/Empty", {});
  };

  const onStartRecordingVideoClick = () => {
    callService(`${nodeName}/start_recording_video`, "std_srvs/Empty", {});
  };

  const onStopRecordingVideoClick = () => {
    callService(`${nodeName}/stop_recording_video`, "std_srvs/Empty", {});
  };

  const onSaveImageClick = () => {
    callService(`${nodeName}/save_image`, "std_srvs/Empty", {});
  };

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Camera (${nodeName}): ${JSON.stringify(
          nodeState
        )}`}</p>
      </div>
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
        {nodeState.laser_detection_enabled ? (
          <Button disabled={!rosConnected} onClick={onStopLaserDetectionClick}>
            Stop Laser Detection
          </Button>
        ) : (
          <Button disabled={!rosConnected} onClick={onStartLaserDetectionClick}>
            Start Laser Detection
          </Button>
        )}
        {nodeState.runner_detection_enabled ? (
          <Button disabled={!rosConnected} onClick={onStopRunnerDetectionClick}>
            Stop Runner Detection
          </Button>
        ) : (
          <Button
            disabled={!rosConnected}
            onClick={onStartRunnerDetectionClick}
          >
            Start Runner Detection
          </Button>
        )}
      </div>
      <div className="flex flex-row items-center gap-4">
        {nodeState.recording_video ? (
          <Button disabled={!rosConnected} onClick={onStopRecordingVideoClick}>
            Stop Recording Video
          </Button>
        ) : (
          <Button disabled={!rosConnected} onClick={onStartRecordingVideoClick}>
            Start Recording Video
          </Button>
        )}
        <Button disabled={!rosConnected} onClick={onSaveImageClick}>
          Save Image
        </Button>
      </div>
      {frameSrc && <img src={frameSrc} alt="Camera Color Frame" />}
    </div>
  );
}
