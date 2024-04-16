"use client";

import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import useCameraNode from "@/lib/useCameraNode";
import FramePreview from "@/components/camera/frame-preview";

export default function Controls() {
  const ros = useContext(ROSContext);
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select camera node name
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const {
    nodeConnected,
    cameraConnected,
    laserDetectionEnabled,
    runnerDetectionEnabled,
    recordingVideo,
    frameSrc,
    setExposure,
    startLaserDetection,
    stopLaserDetection,
    startRunnerDetection,
    stopRunnerDetection,
    startRecordingVideo,
    stopRecordingVideo,
    saveImage,
  } = useCameraNode(nodeName);
  const [exposureMs, setExposureMs] = useState<string>("0.2");

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
        <p className="text-center">{`Camera (${nodeName}): cameraConnected=${cameraConnected}, laserDetectionEnabled=${laserDetectionEnabled}, runnerDetectionEnabled=${runnerDetectionEnabled}, recordingVideo=${recordingVideo}`}</p>
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
            if (!isNaN(value)) {
              setExposureMs(event.target.value);
            }
          }}
        />
        <Button
          disabled={!rosConnected || !nodeConnected}
          onClick={() => {
            setExposure(Number(exposureMs));
          }}
        >
          Set Exposure
        </Button>
        <Button
          disabled={!rosConnected || !nodeConnected}
          onClick={() => {
            setExposure(-1.0);
          }}
        >
          Auto Exposure
        </Button>
      </div>
      <div className="flex flex-row items-center gap-4">
        {laserDetectionEnabled ? (
          <Button
            disabled={!rosConnected || !nodeConnected}
            onClick={() => {
              stopLaserDetection();
            }}
          >
            Stop Laser Detection
          </Button>
        ) : (
          <Button
            disabled={!rosConnected || !nodeConnected}
            onClick={() => {
              startLaserDetection();
            }}
          >
            Start Laser Detection
          </Button>
        )}
        {runnerDetectionEnabled ? (
          <Button
            disabled={!rosConnected || !nodeConnected}
            onClick={() => {
              stopRunnerDetection();
            }}
          >
            Stop Runner Detection
          </Button>
        ) : (
          <Button
            disabled={!rosConnected || !nodeConnected}
            onClick={() => {
              startRunnerDetection();
            }}
          >
            Start Runner Detection
          </Button>
        )}
      </div>
      <div className="flex flex-row items-center gap-4">
        {recordingVideo ? (
          <Button
            disabled={!rosConnected || !nodeConnected}
            onClick={() => {
              stopRecordingVideo();
            }}
          >
            Stop Recording Video
          </Button>
        ) : (
          <Button
            disabled={!rosConnected || !nodeConnected}
            onClick={() => {
              startRecordingVideo();
            }}
          >
            Start Recording Video
          </Button>
        )}
        <Button
          disabled={!rosConnected || !nodeConnected}
          onClick={() => {
            saveImage();
          }}
        >
          Save Image
        </Button>
      </div>
      <FramePreview frameSrc={frameSrc} />
    </div>
  );
}
