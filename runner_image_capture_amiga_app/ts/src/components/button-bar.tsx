import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

enum CaptureMode {
  MANUAL,
  INTERVAL,
  OVERLAP,
}

export default function ButtonBar() {
  const [saveDir, setSaveDir] = useState<string>("~/Pictures/runners");
  const [captureMode, setCaptureMode] = useState<CaptureMode>(
    CaptureMode.MANUAL
  );

  const onSaveDirChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSaveDir(e.target.value);
    },
    []
  );

  const onModeClick = useCallback((captureMode: CaptureMode) => {
    setCaptureMode(captureMode);
  }, []);

  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor="saveDir">Save Directory:</Label>
      <Input
        className="w-96"
        type="text"
        id="saveDir"
        placeholder="Path where images will be saved"
        value={saveDir}
        onChange={onSaveDirChange}
      />
      <Label>Select Mode:</Label>
      <div className="flex flex-row gap-1">
        <Button
          disabled={captureMode === CaptureMode.MANUAL}
          onClick={() => onModeClick(CaptureMode.MANUAL)}
        >
          Manual
        </Button>
        <Button
          disabled={captureMode === CaptureMode.INTERVAL}
          onClick={() => onModeClick(CaptureMode.INTERVAL)}
        >
          Interval
        </Button>
        <Button
          disabled={captureMode === CaptureMode.OVERLAP}
          onClick={() => onModeClick(CaptureMode.OVERLAP)}
        >
          Overlap
        </Button>
      </div>
      {captureMode === CaptureMode.MANUAL ? (
        <ManualMode saveDir={saveDir} />
      ) : null}
    </div>
  );
}

function ManualMode({ saveDir }: { saveDir: string }) {
  const [message, setMessage] = useState<string>("");

  const onCaptureClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/manual`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ saveDir }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      const rawData = await response.json();
      console.log(rawData.file);
      setMessage(rawData.file);
    } catch (error) {
      console.error("Error requesting capture:", error);
    }
  }, [saveDir, setMessage]);

  return (
    <>
      <Label>Manual Mode:</Label>
      <Button onClick={onCaptureClick}>Capture</Button>
      <p className="text-xs">Saved: {message}</p>
    </>
  );
}
