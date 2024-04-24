"use client";

import FramePreview from "@/components/camera/frame-preview";
import NodeCards from "@/components/nodes/node-cards";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import { useMemo, useState } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const {
    nodeInfo,
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
  const [exposureUs, setExposureUs] = useState<string>("200");

  const nodeInfos = useMemo(() => {
    return [rosbridgeNodeInfo, nodeInfo];
  }, [rosbridgeNodeInfo, nodeInfo]);
  const disableButtons = !rosbridgeNodeInfo.connected || !nodeInfo.connected;

  return (
    <div className="flex flex-col gap-4 items-center">
      <NodeCards nodeInfos={nodeInfos} />
      <div className="flex flex-row items-center gap-4">
        <Label className="flex-none w-16" htmlFor="exposure">
          Exposure (us):
        </Label>
        <Input
          className="flex-none w-20"
          type="number"
          id="exposure"
          name="exposure"
          step={10}
          value={exposureUs.toString()}
          onChange={(event) => {
            const value = Number(event.target.value);
            if (!isNaN(value)) {
              setExposureUs(event.target.value);
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
            setExposure(-1.0);
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
      </div>
      <FramePreview frameSrc={frameSrc} />
    </div>
  );
}
