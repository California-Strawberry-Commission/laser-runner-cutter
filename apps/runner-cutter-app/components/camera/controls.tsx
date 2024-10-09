"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import { Button } from "@/components/ui/button";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import { useEffect, useState } from "react";

export default function Controls() {
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/camera0");
  const cameraNode = useCameraNode(nodeName);
  const [exposureUs, setExposureUs] = useState<string>("0");
  const [gainDb, setGainDb] = useState<string>("0");
  const [saveDir, setSaveDir] = useState<string>("");
  const [intervalSecs, setIntervalSecs] = useState<string>("5");

  const disableButtons = !cameraNode.connected;

  // Sync text inputs to node state
  useEffect(() => {
    setExposureUs(String(cameraNode.state.exposureUs));
    setGainDb(String(cameraNode.state.gainDb));
    setSaveDir(cameraNode.state.saveDirectory);
    setIntervalSecs(String(cameraNode.state.imageCaptureIntervalSecs));
  }, [
    setExposureUs,
    cameraNode.state.exposureUs,
    setGainDb,
    cameraNode.state.gainDb,
    setSaveDir,
    cameraNode.state.saveDirectory,
    setIntervalSecs,
    cameraNode.state.imageCaptureIntervalSecs,
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
            helper_text={`Range: [${cameraNode.state.exposureUsRange[0]}, ${cameraNode.state.exposureUsRange[1]}]. Auto: -1`}
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
              cameraNode.setExposure(Number(exposureUs));
            }}
          >
            Set
          </Button>
          <Button
            className="rounded-l-none"
            disabled={disableButtons}
            onClick={() => {
              cameraNode.autoExposure();
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
            helper_text={`Range: [${cameraNode.state.gainDbRange[0]}, ${cameraNode.state.gainDbRange[1]}]. Auto: -1`}
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
              cameraNode.setGain(Number(gainDb));
            }}
          >
            Set
          </Button>
          <Button
            className="rounded-l-none"
            disabled={disableButtons}
            onClick={() => {
              cameraNode.autoGain();
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
              cameraNode.setSaveDirectory(saveDir);
            }}
          >
            Set
          </Button>
        </div>
      </div>
      <div className="flex flex-row items-center gap-4">
        {cameraNode.state.recordingVideo ? (
          <Button
            disabled={disableButtons}
            onClick={() => {
              cameraNode.stopRecordingVideo();
            }}
          >
            Stop Recording Video
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
            onClick={() => {
              cameraNode.startRecordingVideo();
            }}
          >
            Start Recording Video
          </Button>
        )}
        <Button
          disabled={disableButtons}
          onClick={() => {
            cameraNode.saveImage();
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
          {cameraNode.state.intervalCaptureActive ? (
            <Button
              className="rounded-l-none"
              disabled={disableButtons}
              onClick={() => {
                cameraNode.stopIntervalCapture();
              }}
            >
              Stop Interval Capture
            </Button>
          ) : (
            <Button
              className="rounded-l-none"
              disabled={disableButtons}
              onClick={() => {
                cameraNode.startIntervalCapture(Number(intervalSecs));
              }}
            >
              Start Interval Capture
            </Button>
          )}
        </div>
      </div>
      <div className="flex flex-row items-center gap-4">
        {cameraNode.state.laserDetectionEnabled ? (
          <Button
            disabled={disableButtons}
            onClick={() => {
              cameraNode.stopLaserDetection();
            }}
          >
            Stop Laser Detection
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
            onClick={() => {
              cameraNode.startLaserDetection();
            }}
          >
            Start Laser Detection
          </Button>
        )}
        {cameraNode.state.runnerDetectionEnabled ? (
          <Button
            disabled={disableButtons}
            onClick={() => {
              cameraNode.stopRunnerDetection();
            }}
          >
            Stop Runner Detection
          </Button>
        ) : (
          <Button
            disabled={disableButtons}
            onClick={() => {
              cameraNode.startRunnerDetection();
            }}
          >
            Start Runner Detection
          </Button>
        )}
      </div>
      <FramePreviewWithOverlay
        className="w-full h-[520px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.Streaming
        }
      />
    </div>
  );
}
