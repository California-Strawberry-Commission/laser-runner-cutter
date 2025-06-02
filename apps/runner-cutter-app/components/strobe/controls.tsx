"use client";

import DeviceCard, {
  DeviceState,
  convertCameraNodeDeviceState,
} from "@/components/runner-cutter/device-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useCameraNode, { CaptureMode } from "@/lib/useCameraNode";
import { useEffect, useRef, useState } from "react";

export default function Controls({
  cameraNodeName,
}: {
  cameraNodeName: string;
}) {
  const cameraNode = useCameraNode(cameraNodeName);
  const [exposureUs, setExposureUs] = useState<number>(0.0);
  const [gainDb, setGainDb] = useState<number>(0.0);
  const [saveDir, setSaveDir] = useState<string>("");

  const imgRef = useRef<HTMLImageElement>(null);

  const cameraDeviceState = convertCameraNodeDeviceState(cameraNode);
  const disableButtons = cameraDeviceState !== DeviceState.CONNECTED;

  // Sync text inputs to node state
  useEffect(() => {
    setExposureUs(cameraNode.state.exposureUs);
    setGainDb(cameraNode.state.gainDb);
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
              onClick={async () => {
                const res = await cameraNode.acquireSingleFrame();
                const imgElement = imgRef.current;
                if (!imgElement) {
                  return;
                }
                // sensor_msgs/CompressedImage data is base64 encoded
                imgElement.src = `data:image/${res.format};base64,${res.data}`;
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
      <img
        ref={imgRef}
        className="object-contain bg-black w-full h-[480px]"
        alt="Camera Frame Preview"
      />
    </div>
  );
}
