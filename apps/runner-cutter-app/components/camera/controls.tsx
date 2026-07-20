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
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useDetectionNode, { DetectionType } from "@/lib/useDetectionNode";
import { useEffect, useState } from "react";

const detectionTypeLabel: Record<DetectionType, string> = {
  [DetectionType.LASER]: "Laser",
  [DetectionType.RUNNER]: "Runner",
  [DetectionType.CIRCLE]: "Circle",
};

function DetectionToggleButton({
  detectionType,
  detectionNode,
  disabled,
}: {
  detectionType: DetectionType;
  detectionNode: ReturnType<typeof useDetectionNode>;
  disabled: boolean;
}) {
  const label = detectionTypeLabel[detectionType];
  const active =
    detectionNode.state.enabledDetectionTypes.includes(detectionType);
  return (
    <Button
      disabled={disabled}
      onClick={() =>
        active
          ? detectionNode.stopDetection(detectionType)
          : detectionNode.startDetection(detectionType)
      }
    >
      {active ? `Stop ${label} Detection` : `Start ${label} Detection`}
    </Button>
  );
}

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

  // Sync text inputs to node params on connect
  useEffect(
    () => {
      async function fetchParams() {
        if (cameraNode.connected) {
          const exposureUs = await cameraNode.getExposureUs();
          setExposureUs(exposureUs);

          const gainDb = await cameraNode.getGainDb();
          setGainDb(gainDb);

          const saveDir = await cameraNode.getSaveDir();
          setSaveDir(saveDir);

          const imageCaptureIntervalSecs =
            await cameraNode.getImageCaptureIntervalSecs();
          setIntervalSecs(imageCaptureIntervalSecs);
        }
      }

      fetchParams();
    },
    // We intentionally omit cameraNode object to avoid re-running on every
    // render. `.connected` is the signal we care about.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [cameraNode.connected],
  );

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
                    const value = parseFloat(str);
                    if (!isNaN(value)) {
                      setExposureUs(value);
                    }
                  }}
                />
                <Button
                  className="rounded-none"
                  disabled={disableButtons}
                  onClick={() => cameraNode.setExposureUs(exposureUs)}
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
                    const value = parseFloat(str);
                    if (!isNaN(value)) {
                      setGainDb(value);
                    }
                  }}
                />
                <Button
                  className="rounded-none"
                  disabled={disableButtons}
                  onClick={() => cameraNode.setGainDb(gainDb)}
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
                  onChange={setSaveDir}
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
              <Button
                disabled={disableButtons}
                onClick={() =>
                  detectionNode.state.recordingVideo
                    ? detectionNode.stopRecordingVideo()
                    : detectionNode.startRecordingVideo()
                }
              >
                {detectionNode.state.recordingVideo
                  ? "Stop Recording Video"
                  : "Start Recording Video"}
              </Button>
              <Button
                disabled={disableButtons}
                onClick={() => cameraNode.saveImage()}
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
                    const value = parseFloat(str);
                    if (!isNaN(value)) {
                      setIntervalSecs(value);
                    }
                  }}
                />
                <Button
                  className="rounded-l-none"
                  disabled={disableButtons}
                  onClick={() =>
                    cameraNode.state.intervalCaptureActive
                      ? cameraNode.stopIntervalCapture()
                      : cameraNode.startIntervalCapture(intervalSecs)
                  }
                >
                  {cameraNode.state.intervalCaptureActive
                    ? "Stop Interval Capture"
                    : "Start Interval Capture"}
                </Button>
              </div>
            </div>
            <div className="flex flex-row items-center gap-4">
              <DetectionToggleButton
                detectionType={DetectionType.LASER}
                detectionNode={detectionNode}
                disabled={disableButtons}
              />
              <DetectionToggleButton
                detectionType={DetectionType.RUNNER}
                detectionNode={detectionNode}
                disabled={disableButtons}
              />
              <DetectionToggleButton
                detectionType={DetectionType.CIRCLE}
                detectionNode={detectionNode}
                disabled={disableButtons}
              />
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
      <FramePreviewLiveKit
        className="w-full h-[480px]"
        topicName={`${detectionNodeName}/debug/depth_image`}
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        showRotateButton
      />
    </div>
  );
}
