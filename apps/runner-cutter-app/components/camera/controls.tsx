"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import DeviceCard, {
  DeviceState,
  convertCameraNodeDeviceState,
} from "@/components/runner-cutter/device-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useCameraNode, {
  DeviceState as CameraDeviceState,
  DetectionType,
} from "@/lib/useCameraNode";
import { useEffect, useState } from "react";

export default function Controls({
  cameraNodeName,
}: {
  cameraNodeName: string;
}) {
  const cameraNode = useCameraNode(cameraNodeName);
  const [exposureUs, setExposureUs] = useState<number>(0.0);
  const [gainDb, setGainDb] = useState<number>(0.0);
  const [saveDir, setSaveDir] = useState<string>("");
  const [intervalSecs, setIntervalSecs] = useState<number>(5.0);

  const cameraDeviceState = convertCameraNodeDeviceState(cameraNode);
  const disableButtons = cameraDeviceState !== DeviceState.CONNECTED;

  // Sync text inputs to node state
  useEffect(() => {
    setExposureUs(cameraNode.state.exposureUs);
    setGainDb(cameraNode.state.gainDb);
    setSaveDir(cameraNode.state.saveDirectory);
    setIntervalSecs(cameraNode.state.imageCaptureIntervalSecs);
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
      <div className="flex flex-row gap-4 items-center">
        <DeviceCard
          className="h-fit"
          deviceName="Camera"
          deviceState={cameraDeviceState}
          onConnectClick={() => cameraNode.startDevice()}
          onDisconnectClick={() => cameraNode.closeDevice()}
        />
        <Card>
          <CardContent className="p-4 flex flex-col items-center gap-4">
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
                      setExposureUs(value);
                    }
                  }}
                />
                <Button
                  className="rounded-none"
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.setExposure(exposureUs);
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
                      setGainDb(value);
                    }
                  }}
                />
                <Button
                  className="rounded-none"
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.setGain(gainDb);
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
                      setIntervalSecs(value);
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
                      cameraNode.startIntervalCapture(intervalSecs);
                    }}
                  >
                    Start Interval Capture
                  </Button>
                )}
              </div>
            </div>
            <div className="flex flex-row items-center gap-4">
              {cameraNode.state.enabledDetectionTypes.includes(
                DetectionType.LASER
              ) ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.stopDetection(DetectionType.LASER);
                  }}
                >
                  Stop Laser Detection
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.startDetection(DetectionType.LASER);
                  }}
                >
                  Start Laser Detection
                </Button>
              )}
              {cameraNode.state.enabledDetectionTypes.includes(
                DetectionType.RUNNER
              ) ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.stopDetection(DetectionType.RUNNER);
                  }}
                >
                  Stop Runner Detection
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.startDetection(DetectionType.RUNNER);
                  }}
                >
                  Start Runner Detection
                </Button>
              )}
              {cameraNode.state.enabledDetectionTypes.includes(
                DetectionType.CIRCLE
              ) ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.stopDetection(DetectionType.CIRCLE);
                  }}
                >
                  Stop Circle Detection
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.startDetection(DetectionType.CIRCLE);
                  }}
                >
                  Start Circle Detection
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
      <FramePreviewWithOverlay
        className="w-full h-[480px]"
        topicName={`/detection_node/overlay_frame`}
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        showRotateButton
      />
    </div>
  );
}
