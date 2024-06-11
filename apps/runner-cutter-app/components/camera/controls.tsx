"use client";

import FramePreview from "@/components/camera/frame-preview";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import { useState } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const {
    nodeInfo,
    laserDetectionEnabled,
    runnerDetectionEnabled,
    recordingVideo,
    intervalCaptureActive,
    logMessages,
    setExposure,
    autoExposure,
    startLaserDetection,
    stopLaserDetection,
    startRunnerDetection,
    stopRunnerDetection,
    startRecordingVideo,
    stopRecordingVideo,
    startIntervalCapture,
    stopIntervalCapture,
    saveImage,
  } = useCameraNode(nodeName);
  const [exposureUs, setExposureUs] = useState<string>("200");
  const [intervalSecs, setIntervalSecs] = useState<string>("5");

  const disableButtons = !rosbridgeNodeInfo.connected || !nodeInfo.connected;

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row items-center gap-4">
        <Label className="flex-none w-16" htmlFor="exposure">
          Exposure (us):
        </Label>
        <Input
          className="flex-none w-24"
          type="number"
          id="exposure"
          name="exposure"
          step={10}
          value={exposureUs.toString()}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setExposureUs(str);
            }
          }}
        />
        <Button
          disabled={disableButtons}
          onClick={() => {
            setExposure(Number(exposureUs));
          }}
        >
          Set Exposure
        </Button>
        <Button
          disabled={disableButtons}
          onClick={() => {
            autoExposure();
          }}
        >
          Auto Exposure
        </Button>
      </div>
      <div className="flex flex-row items-center gap-4">
        {laserDetectionEnabled ? (
          <Button
            disabled={disableButtons}
            onClick={() => {
              stopLaserDetection();
            }}
          >
            Stop Laser Detection
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
            onClick={() => {
              startLaserDetection();
            }}
          >
            Start Laser Detection
          </Button>
        )}
        {runnerDetectionEnabled ? (
          <Button
            disabled={disableButtons}
            onClick={() => {
              stopRunnerDetection();
            }}
          >
            Stop Runner Detection
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
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
            disabled={disableButtons}
            onClick={() => {
              stopRecordingVideo();
            }}
          >
            Stop Recording Video
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
            onClick={() => {
              startRecordingVideo();
            }}
          >
            Start Recording Video
          </Button>
        )}
        <Button
          disabled={disableButtons}
          onClick={() => {
            saveImage();
          }}
        >
          Save Image
        </Button>
        <Label className="flex-none w-16" htmlFor="interval">
          Interval (s):
        </Label>
        <Input
          className="flex-none w-20"
          type="number"
          id="interval"
          name="interval"
          step={1}
          value={intervalSecs.toString()}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setIntervalSecs(str);
            }
          }}
        />
        {intervalCaptureActive ? (
          <Button
            disabled={disableButtons}
            onClick={() => {
              stopIntervalCapture();
            }}
          >
            Stop Interval Capture
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
            onClick={() => {
              startIntervalCapture(Number(intervalSecs));
            }}
          >
            Start Interval Capture
          </Button>
        )}
      </div>
      <FramePreview topicName={"/camera0/debug_frame"} />
      <div className="w-full">
        {logMessages.map((msg, index) => (
          <p className="text-xs" key={index}>
            {msg}
          </p>
        ))}
      </div>
    </div>
  );
}
