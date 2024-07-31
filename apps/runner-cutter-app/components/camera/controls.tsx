"use client";

import FramePreview from "@/components/camera/frame-preview";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import { useEffect, useState } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const {
    nodeInfo,
    nodeState,
    logMessages,
    setExposure,
    autoExposure,
    setGain,
    autoGain,
    setSaveDirectory,
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
  const [exposureUs, setExposureUs] = useState<string>("0");
  const [gainDb, setGainDb] = useState<string>("0");
  const [saveDir, setSaveDir] = useState<string>("");
  const [intervalSecs, setIntervalSecs] = useState<string>("5");

  const disableButtons = !rosbridgeNodeInfo.connected || !nodeInfo.connected;

  // Sync text inputs to node state
  useEffect(() => {
    setExposureUs(String(nodeState.exposureUs));
    setGainDb(String(nodeState.gainDb));
    setSaveDir(nodeState.saveDirectory);
  }, [
    setExposureUs,
    nodeState.exposureUs,
    setGainDb,
    nodeState.gainDb,
    setSaveDir,
    nodeState.saveDirectory,
  ]);

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row items-center gap-4 mb-4">
        <div className="flex flex-row items-center gap-[1px]">
          <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            name="exposure"
            label="Exposure (µs)"
            helper_text={`Range: [${nodeState.exposureUsRange[0]}, ${nodeState.exposureUsRange[1]}]. Auto: -1`}
            step={10}
            value={exposureUs}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setExposureUs(str);
              }
            }}
          />
          <Button
            className="rounded-none"
            disabled={disableButtons}
            onClick={() => {
              setExposure(Number(exposureUs));
            }}
          >
            Set
          </Button>
          <Button
            className="rounded-l-none"
            disabled={disableButtons}
            onClick={() => {
              autoExposure();
            }}
          >
            Auto
          </Button>
        </div>
        <div className="flex flex-row items-center gap-[1px]">
          <InputWithLabel
            className="flex-none w-20 rounded-r-none"
            type="number"
            id="gain"
            name="gain"
            label="Gain (dB)"
            helper_text={`Range: [${nodeState.gainDbRange[0]}, ${nodeState.gainDbRange[1]}]. Auto: -1`}
            step={1}
            value={gainDb}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setGainDb(str);
              }
            }}
          />
          <Button
            className="rounded-none"
            disabled={disableButtons}
            onClick={() => {
              setGain(Number(gainDb));
            }}
          >
            Set
          </Button>
          <Button
            className="rounded-l-none"
            disabled={disableButtons}
            onClick={() => {
              autoGain();
            }}
          >
            Auto
          </Button>
        </div>
        <div className="flex flex-row items-center">
          <InputWithLabel
            className="flex-none w-64 rounded-r-none"
            type="text"
            id="saveDir"
            name="saveDir"
            label="Save Directory"
            value={saveDir}
            onChange={(str) => {
              setSaveDir(str);
            }}
          />
          <Button
            className="rounded-l-none"
            disabled={disableButtons}
            onClick={() => {
              setSaveDirectory(saveDir);
            }}
          >
            Set
          </Button>
        </div>
      </div>
      <div className="flex flex-row items-center gap-4">
        {nodeState.recordingVideo ? (
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
        <div className="flex flex-row items-center">
          <InputWithLabel
            className="flex-none w-20 rounded-r-none"
            type="number"
            id="interval"
            name="interval"
            label="Interval (s)"
            step={1}
            value={intervalSecs}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setIntervalSecs(str);
              }
            }}
          />
          {nodeState.intervalCaptureActive ? (
            <Button
              className="rounded-l-none"
              disabled={disableButtons}
              onClick={() => {
                stopIntervalCapture();
              }}
            >
              Stop Interval Capture
            </Button>
          ) : (
            <Button
              className="rounded-l-none"
              disabled={disableButtons}
              onClick={() => {
                startIntervalCapture(Number(intervalSecs));
              }}
            >
              Start Interval Capture
            </Button>
          )}
        </div>
      </div>
      <div className="flex flex-row items-center gap-4">
        {nodeState.laserDetectionEnabled ? (
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
        {nodeState.runnerDetectionEnabled ? (
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
      <FramePreview height={520} topicName={"/camera0/debug_frame"} />
      <Popover>
        <PopoverTrigger asChild>
          <Button className="fixed bottom-4 right-4">Show Logs</Button>
        </PopoverTrigger>
        <PopoverContent className="m-4 w-96 bg-black bg-opacity-70 border-0">
          {logMessages.map((msg, index) => (
            <p className="text-xs text-white" key={index}>
              {msg}
            </p>
          ))}
        </PopoverContent>
      </Popover>
    </div>
  );
}
