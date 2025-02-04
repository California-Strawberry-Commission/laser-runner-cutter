"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import DeviceCard, {
  DeviceState,
} from "@/components/runner-cutter/device-card";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useCameraNode, {
  CaptureMode,
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

  let cameraDeviceState = DeviceState.UNAVAILABLE;
  if (cameraNode.connected) {
    switch (cameraNode.state.deviceState) {
      case CameraDeviceState.DISCONNECTED:
        cameraDeviceState = DeviceState.DISCONNECTED;
        break;
      case CameraDeviceState.CONNECTING:
        cameraDeviceState = DeviceState.CONNECTING;
        break;
      case CameraDeviceState.STREAMING:
        cameraDeviceState = DeviceState.CONNECTED;
        break;
      default:
        break;
    }
  }

  const disableButtons = cameraDeviceState !== DeviceState.CONNECTED;

  // Sync text inputs to node state
  useEffect(() => {
    setExposureUs(String(cameraNode.state.exposureUs));
    setGainDb(String(cameraNode.state.gainDb));
    setSaveDir(cameraNode.state.saveDirectory);
  }, [
    setExposureUs,
    cameraNode.state.exposureUs,
    setGainDb,
    cameraNode.state.gainDb,
    setSaveDir,
    cameraNode.state.saveDirectory,
  ]);

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row gap-4 items-center">
        <DeviceCard
          className="h-fit"
          deviceName="Camera"
          deviceState={cameraDeviceState}
          onConnectClick={() =>
            cameraNode.startDevice(CaptureMode.SINGLE_FRAME)
          }
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
            </div>
            <div className="flex flex-row items-center gap-4">
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
          </CardContent>
        </Card>
        <Card
          className={
            cameraDeviceState === DeviceState.CONNECTED
              ? "bg-green-500"
              : "bg-gray-300"
          }
        >
          <CardHeader className="p-4">
            <CardTitle className="text-lg">Capture</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0 flex flex-row gap-4">
            <Button
              disabled={disableButtons}
              onClick={() => {
                cameraNode.acquireSingleFrame();
              }}
            >
              Acquire Frame
            </Button>
            <Button
              disabled={disableButtons}
              onClick={() => {
                cameraNode.saveImage();
              }}
            >
              Save Image
            </Button>
          </CardContent>
        </Card>
      </div>
      <FramePreviewWithOverlay
        className="w-full h-[480px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
      />
    </div>
  );
}
