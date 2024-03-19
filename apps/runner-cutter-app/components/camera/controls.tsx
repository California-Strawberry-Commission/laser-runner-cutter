"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import useCameraNode from "@/lib/useCameraNode";

export default function Controls() {
  const { ros } = useROS();
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select camera node name
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const {
    connected,
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
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Camera (${nodeName}): connected=${connected}, laserDetectionEnabled=${laserDetectionEnabled}, runnerDetectionEnabled=${runnerDetectionEnabled}, recordingVideo=${recordingVideo}`}</p>
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
          disabled={!rosConnected}
          onClick={() => {
            setExposure(Number(exposureMs));
          }}
        >
          Set Exposure
        </Button>
        <Button
          disabled={!rosConnected}
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
            disabled={!rosConnected}
            onClick={() => {
              stopLaserDetection();
            }}
          >
            Stop Laser Detection
          </Button>
        ) : (
          <Button
            disabled={!rosConnected}
            onClick={() => {
              startLaserDetection();
            }}
          >
            Start Laser Detection
          </Button>
        )}
        {runnerDetectionEnabled ? (
          <Button
            disabled={!rosConnected}
            onClick={() => {
              stopRunnerDetection();
            }}
          >
            Stop Runner Detection
          </Button>
        ) : (
          <Button
            disabled={!rosConnected}
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
            disabled={!rosConnected}
            onClick={() => {
              stopRecordingVideo();
            }}
          >
            Stop Recording Video
          </Button>
        ) : (
          <Button
            disabled={!rosConnected}
            onClick={() => {
              startRecordingVideo();
            }}
          >
            Start Recording Video
          </Button>
        )}
        <Button
          disabled={!rosConnected}
          onClick={() => {
            saveImage();
          }}
        >
          Save Image
        </Button>
      </div>
      {frameSrc && <img src={frameSrc} alt="Camera Color Frame" />}
    </div>
  );
}
