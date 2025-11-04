"use client";

import FramePreviewLiveKit from "@/components/camera/frame-preview-livekit";
import DeviceCard, {
  DeviceState,
  convertCameraNodeDeviceState,
} from "@/components/runner-cutter/device-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useCameraNode, {
  DeviceState as CameraDeviceState
} from "@/lib/useCameraNode";
import useDetectionNode, {DetectionType} from "@/lib/useDetectionNode";
import { useEffect, useState } from "react";

export default function Controls({
  cameraNodeName,
  detectionNodeName,
}: {
  cameraNodeName: string;
  detectionNodeName: string;
}) {
  const cameraNode = useCameraNode(cameraNodeName);
  const detectionNode = useDetectionNode(detectionNodeName);
  const [exposureUs, setExposureUs] = useState<number>(0.0);
  const [gainDb, setGainDb] = useState<number>(0.0);
  const [saveDir, setSaveDir] = useState<string>("");
  const [intervalSecs, setIntervalSecs] = useState<number>(5.0);

  const cameraDeviceState = convertCameraNodeDeviceState(cameraNode);
  const disableButtons = cameraDeviceState !== DeviceState.CONNECTED;

  // Sync text inputs to node params
  useEffect(() => {
    async function fetchParams() {
      if (cameraNode.connected) {
        const exposureUs = await cameraNode.getExposureUs();
        setExposureUs(exposureUs);

        const gainDb = await cameraNode.getGainDb();
        setGainDb(gainDb);

        const saveDir = await cameraNode.getSaveDir();
        setSaveDir(saveDir);

        const imageCaptureIntervalSecs = await cameraNode.getImageCaptureIntervalSecs();
        setIntervalSecs(imageCaptureIntervalSecs);
      }
    }

    fetchParams();
  },
  // We intentionally did not add cameraNode to deps
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [
    cameraNode.connected,
    setExposureUs,
    setGainDb,
    setSaveDir,
    setIntervalSecs,
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
                    cameraNode.setExposureUs(exposureUs);
                  }}
                >
                  Set
                </Button>
                <Button
                  className="rounded-l-none"
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.setExposureUs(-1);
                    setExposureUs(-1);
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
                    cameraNode.setGainDb(gainDb);
                  }}
                >
                  Set
                </Button>
                <Button
                  className="rounded-l-none"
                  disabled={disableButtons}
                  onClick={() => {
                    cameraNode.setGainDb(-1);
                    setGainDb(-1);
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
                    cameraNode.setSaveDir(saveDir);
                    detectionNode.setSaveDir(saveDir);
                  }}
                >
                  Set
                </Button>
              </div>
            </div>
            <div className="flex flex-row items-center gap-4">
              {detectionNode.state.recordingVideo ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.stopRecordingVideo();
                  }}
                >
                  Stop Recording Video
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.startRecordingVideo();
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
              {detectionNode.state.enabledDetectionTypes.includes(
                DetectionType.LASER
              ) ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.stopDetection(DetectionType.LASER);
                  }}
                >
                  Stop Laser Detection
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.startDetection(DetectionType.LASER);
                  }}
                >
                  Start Laser Detection
                </Button>
              )}
              {detectionNode.state.enabledDetectionTypes.includes(
                DetectionType.RUNNER
              ) ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.stopDetection(DetectionType.RUNNER);
                  }}
                >
                  Stop Runner Detection
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.startDetection(DetectionType.RUNNER);
                  }}
                >
                  Start Runner Detection
                </Button>
              )}
              {detectionNode.state.enabledDetectionTypes.includes(
                DetectionType.CIRCLE
              ) ? (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.stopDetection(DetectionType.CIRCLE);
                  }}
                >
                  Stop Circle Detection
                </Button>
              ) : (
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    detectionNode.startDetection(DetectionType.CIRCLE);
                  }}
                >
                  Start Circle Detection
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
      <FramePreviewLiveKit
        className="w-full h-[480px]"
        topicName={`${detectionNodeName}/debug/image`}
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        showRotateButton
      />
    </div>
  );
}
