"use client";

import ColorPicker from "@/components/laser/color-picker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useRgbColor from "@/lib/useRgbColor";
import { useEffect, useState } from "react";
import { toast } from "sonner";

export default function Controls({
  cameraNodeName,
  controlNodeName,
}: {
  cameraNodeName: string;
  controlNodeName: string;
}) {
  const cameraNode = useCameraNode(cameraNodeName);
  const controlNode = useControlNode(controlNodeName);

  const [paramsFetched, setParamsFetched] = useState<boolean>(false);
  const [dirty, setDirty] = useState<boolean>(false);
  const [exposureUs, setExposureUs] = useState<number>(0.0);
  const [gainDb, setGainDb] = useState<number>(0.0);
  const [calibrationGridSize, setCalibrationGridSize] = useState<{
    numX: number;
    numY: number;
  }>({ numX: 0, numY: 0 });
  const [calibrationXBounds, setCalibrationXBounds] = useState<{
    min: number;
    max: number;
  }>({ min: 0.0, max: 1.0 });
  const [calibrationYBounds, setCalibrationYBounds] = useState<{
    min: number;
    max: number;
  }>({ min: 0.0, max: 1.0 });
  const [trackingLaserColor, setTrackingLaserColor] = useRgbColor({
    r: 0.0,
    g: 0.0,
    b: 0.0,
  });
  const [burnLaserColor, setBurnLaserColor] = useRgbColor({
    r: 0.0,
    g: 0.0,
    b: 0.0,
  });
  const [burnTimeSecs, setBurnTimeSecs] = useState<number>(0.0);
  const [enableAiming, setEnableAiming] = useState<boolean>(false);
  const [targetAttempts, setTargetAttempts] = useState<number>(0);
  const [autoDisarmSecs, setAutoDisarmSecs] = useState<number>(0.0);
  const [saveDir, setSaveDir] = useState<string>("");

  // Sync inputs to node params
  useEffect(() => {
    async function fetchParams() {
      if (cameraNode.connected) {
        setExposureUs(cameraNode.state.exposureUs);
        setGainDb(cameraNode.state.gainDb);
      }

      if (controlNode.connected) {
        const calibrationGridSizeRes =
          await controlNode.getCalibrationGridSize();
        if (calibrationGridSizeRes.length >= 2) {
          setCalibrationGridSize({
            numX: calibrationGridSizeRes[0],
            numY: calibrationGridSizeRes[1],
          });
        }

        const calibrationXBoundsRes = await controlNode.getCalibrationXBounds();
        if (calibrationXBoundsRes.length >= 2) {
          setCalibrationXBounds({
            min: calibrationXBoundsRes[0],
            max: calibrationXBoundsRes[1],
          });
        }

        const calibrationYBoundsRes = await controlNode.getCalibrationYBounds();
        if (calibrationYBoundsRes.length >= 2) {
          setCalibrationYBounds({
            min: calibrationYBoundsRes[0],
            max: calibrationYBoundsRes[1],
          });
        }

        const trackingLaserColorRes = await controlNode.getTrackingLaserColor();
        if (trackingLaserColorRes.length >= 3) {
          setTrackingLaserColor({
            r: trackingLaserColorRes[0],
            g: trackingLaserColorRes[1],
            b: trackingLaserColorRes[2],
          });
        }

        const burnLaserColorRes = await controlNode.getBurnLaserColor();
        if (burnLaserColorRes.length >= 3) {
          setBurnLaserColor({
            r: burnLaserColorRes[0],
            g: burnLaserColorRes[1],
            b: burnLaserColorRes[2],
          });
        }

        const burnTimeSecsRes = await controlNode.getBurnTimeSecs();
        setBurnTimeSecs(burnTimeSecsRes);

        const enableAimingRes = await controlNode.getEnableAiming();
        setEnableAiming(enableAimingRes);

        const targetAttemptsRes = await controlNode.getTargetAttempts();
        if (Number.isInteger(targetAttemptsRes)) {
          setTargetAttempts(targetAttemptsRes);
        }

        const autoDisarmSecsRes = await controlNode.getAutoDisarmSecs();
        setAutoDisarmSecs(autoDisarmSecsRes);

        const saveDirRes = await controlNode.getSaveDir();
        setSaveDir(saveDirRes);

        setParamsFetched(true);
      }
    }

    fetchParams();
    // We intentionally did not add controlNode to deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    cameraNode.connected,
    controlNode.connected,
    cameraNode.state.exposureUs,
    cameraNode.state.gainDb,
    setExposureUs,
    setGainDb,
    setCalibrationGridSize,
    setCalibrationXBounds,
    setCalibrationYBounds,
    setTrackingLaserColor,
    setBurnLaserColor,
    setBurnTimeSecs,
    setEnableAiming,
    setTargetAttempts,
    setAutoDisarmSecs,
    setSaveDir,
    setParamsFetched,
  ]);

  const enableFields = controlNode.connected && paramsFetched;
  const enableSaveButton = controlNode.connected && dirty;

  const handleSave = () => {
    cameraNode.setExposure(exposureUs);
    cameraNode.setGain(gainDb);
    controlNode.setCalibrationGridSize(
      calibrationGridSize.numX,
      calibrationGridSize.numY
    );
    controlNode.setCalibrationXBounds(
      calibrationXBounds.min,
      calibrationXBounds.max
    );
    controlNode.setCalibrationYBounds(
      calibrationYBounds.min,
      calibrationYBounds.max
    );
    controlNode.setTrackingLaserColor(
      trackingLaserColor.r,
      trackingLaserColor.g,
      trackingLaserColor.b
    );
    controlNode.setBurnLaserColor(
      burnLaserColor.r,
      burnLaserColor.g,
      burnLaserColor.b
    );
    controlNode.setBurnTimeSecs(burnTimeSecs);
    controlNode.setEnableAiming(enableAiming);
    controlNode.setTargetAttempts(targetAttempts);
    controlNode.setAutoDisarmSecs(autoDisarmSecs);
    controlNode.setSaveDir(saveDir);
    toast.info("Settings saved successfully.");
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <div className="flex flex-col gap-1">
          <Label>Exposure (µs)</Label>
          <Label className="text-xs font-light">{`${
            cameraNode.state.deviceState !== CameraDeviceState.STREAMING
              ? "Camera not connected. "
              : ""
          }Range: [${cameraNode.state.exposureUsRange[0]}, ${
            cameraNode.state.exposureUsRange[1]
          }]. Auto: -1`}</Label>
        </div>
        <Input
          type="number"
          inputMode="numeric"
          id="exposure"
          name="exposure"
          step={10}
          disabled={!enableFields}
          value={exposureUs}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setExposureUs(value);
              setDirty(true);
            }
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <div className="flex flex-col gap-1">
          <Label>Gain (dB)</Label>
          <Label className="text-xs font-light">{`${
            cameraNode.state.deviceState !== CameraDeviceState.STREAMING
              ? "Camera not connected. "
              : ""
          }Range: [${cameraNode.state.gainDbRange[0]}, ${
            cameraNode.state.gainDbRange[1]
          }]. Auto: -1`}</Label>
        </div>
        <Input
          type="number"
          inputMode="numeric"
          id="gain"
          name="gain"
          step={1}
          disabled={!enableFields}
          value={gainDb}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setGainDb(value);
              setDirty(true);
            }
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <div className="flex flex-col gap-1">
          <Label>Calibration grid size</Label>
          <Label className="text-xs font-light">
            (# x divisions, # y divisions)
          </Label>
        </div>
        <div className="flex flex-row gap-2">
          <Input
            type="number"
            inputMode="numeric"
            id="calibrationGridSizeX"
            name="calibrationGridSizeX"
            step={1}
            disabled={!enableFields}
            value={calibrationGridSize.numX}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setCalibrationGridSize({
                  numX: Math.floor(value),
                  numY: calibrationGridSize.numY,
                });
                setDirty(true);
              }
            }}
          />
          <Input
            type="number"
            inputMode="numeric"
            id="calibrationGridSizeY"
            name="calibrationGridSizeY"
            step={1}
            disabled={!enableFields}
            value={calibrationGridSize.numY}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setCalibrationGridSize({
                  numX: calibrationGridSize.numX,
                  numY: Math.floor(value),
                });
                setDirty(true);
              }
            }}
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <div className="flex flex-col gap-1">
          <Label>Calibration bounds</Label>
          <Label className="text-xs font-light">
            (min x, max x, min y, max y)
          </Label>
        </div>
        <div className="flex flex-row gap-2">
          <Input
            type="number"
            inputMode="numeric"
            id="calibrationXBoundsMin"
            name="calibrationXBoundsMin"
            step={0.1}
            disabled={!enableFields}
            value={calibrationXBounds.min}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setCalibrationXBounds({
                  min: value,
                  max: calibrationXBounds.max,
                });
                setDirty(true);
              }
            }}
          />
          <Input
            type="number"
            inputMode="numeric"
            id="calibrationXBoundsMax"
            name="calibrationXBoundsMax"
            step={0.1}
            disabled={!enableFields}
            value={calibrationXBounds.max}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setCalibrationXBounds({
                  min: calibrationXBounds.min,
                  max: value,
                });
                setDirty(true);
              }
            }}
          />
          <Input
            type="number"
            inputMode="numeric"
            id="calibrationYBoundsMin"
            name="calibrationYBoundsMin"
            step={0.1}
            disabled={!enableFields}
            value={calibrationYBounds.min}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setCalibrationYBounds({
                  min: value,
                  max: calibrationYBounds.max,
                });
                setDirty(true);
              }
            }}
          />
          <Input
            type="number"
            inputMode="numeric"
            id="calibrationYBoundsMax"
            name="calibrationYBoundsMax"
            step={0.1}
            disabled={!enableFields}
            value={calibrationYBounds.max}
            onChange={(str) => {
              const value = Number(str);
              if (!isNaN(value)) {
                setCalibrationYBounds({
                  min: calibrationYBounds.min,
                  max: value,
                });
                setDirty(true);
              }
            }}
          />
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Tracking laser color</Label>
        <ColorPicker
          disabled={!enableFields}
          color={trackingLaserColor}
          onColorChange={(color) => {
            setTrackingLaserColor(color);
            setDirty(true);
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Burn laser color</Label>
        <ColorPicker
          disabled={!enableFields}
          color={burnLaserColor}
          onColorChange={(color) => {
            setBurnLaserColor(color);
            setDirty(true);
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Burn time (s)</Label>
        <Input
          type="number"
          inputMode="numeric"
          id="burnTimeSecs"
          name="burnTimeSecs"
          step={1}
          disabled={!enableFields}
          value={burnTimeSecs}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setBurnTimeSecs(value);
              setDirty(true);
            }
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Enable aiming</Label>
        <Switch
          id="enableAiming"
          disabled={!enableFields}
          checked={enableAiming}
          onCheckedChange={(checked) => {
            setEnableAiming(checked);
            setDirty(true);
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Target attempts</Label>
        <Input
          type="number"
          inputMode="numeric"
          id="targetAttempts"
          name="targetAttempts"
          step={1}
          disabled={!enableFields}
          value={targetAttempts}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setTargetAttempts(Math.floor(value));
              setDirty(true);
            }
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Auto disarm time (s)</Label>
        <Input
          type="number"
          inputMode="numeric"
          id="autoDisarmSecs"
          name="autoDisarmSecs"
          step={1}
          disabled={!enableFields}
          value={autoDisarmSecs}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setAutoDisarmSecs(value);
              setDirty(true);
            }
          }}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-2 w-full">
        <Label>Save directory</Label>
        <Input
          type="text"
          inputMode="text"
          id="saveDir"
          name="saveDir"
          disabled={!enableFields}
          value={saveDir}
          onChange={(str) => {
            setSaveDir(str);
            setDirty(true);
          }}
        />
      </div>
      <Button
        className="w-32"
        disabled={!enableSaveButton}
        onClick={handleSave}
      >
        Save Changes
      </Button>
    </div>
  );
}
