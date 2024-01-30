import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";

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
  const [captureInProgress, setCaptureInProgress] = useState<boolean>(false);

  const onSaveDirChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSaveDir(e.target.value);
    },
    []
  );

  const onModeClick = useCallback((captureMode: CaptureMode) => {
    setCaptureMode(captureMode);
  }, []);

  const onCaptureStateChange = useCallback((inProgress: boolean) => {
    setCaptureInProgress(inProgress);
  }, []);

  return (
    <div className="flex flex-col gap-2 w-96">
      <div className="flex flex-row gap-2 items-center">
        <Label htmlFor="saveDir">Save Directory:</Label>
        <Input
          type="text"
          id="saveDir"
          placeholder="Path where images will be saved"
          value={saveDir}
          onChange={onSaveDirChange}
        />
      </div>
      <div className="flex flex-row gap-2 items-center">
        <Label>Select Mode:</Label>
        <div className="flex flex-row gap-1">
          <Button
            disabled={captureMode === CaptureMode.MANUAL || captureInProgress}
            onClick={() => onModeClick(CaptureMode.MANUAL)}
          >
            Manual
          </Button>
          <Button
            disabled={captureMode === CaptureMode.INTERVAL || captureInProgress}
            onClick={() => onModeClick(CaptureMode.INTERVAL)}
          >
            Interval
          </Button>
          <Button
            disabled={captureMode === CaptureMode.OVERLAP || captureInProgress}
            onClick={() => onModeClick(CaptureMode.OVERLAP)}
          >
            Overlap
          </Button>
        </div>
      </div>
      <Separator />
      {captureMode === CaptureMode.MANUAL ? (
        <ManualMode saveDir={saveDir} />
      ) : null}
      {captureMode === CaptureMode.INTERVAL ? (
        <IntervalMode
          saveDir={saveDir}
          captureInProgress={captureInProgress}
          onCaptureStateChange={onCaptureStateChange}
        />
      ) : null}
      {captureMode === CaptureMode.OVERLAP ? (
        <OverlapMode
          saveDir={saveDir}
          captureInProgress={captureInProgress}
          onCaptureStateChange={onCaptureStateChange}
        />
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
      setMessage(`Saved: ${rawData.file}`);
    } catch (error) {
      console.error("Error requesting capture:", error);
    }
  }, [saveDir, setMessage]);

  return (
    <>
      <h2 className="text-center font-bold">Manual Mode</h2>
      <Button onClick={onCaptureClick}>Capture</Button>
      <p className="text-xs">{message}</p>
    </>
  );
}

function IntervalMode({
  saveDir,
  captureInProgress,
  onCaptureStateChange,
}: {
  saveDir: string;
  captureInProgress: boolean;
  onCaptureStateChange: (inProgress: boolean) => void;
}) {
  const [intervalSecs, setIntervalSecs] = useState<number>(5);

  const onIntervalChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(e.target.value);
      if (isNaN(value) || value < 0) {
        return;
      }
      setIntervalSecs(value);
    },
    []
  );

  const onStartClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/interval`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ intervalSecs, saveDir }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      const rawData = await response.json();
      console.log(rawData.file);
      onCaptureStateChange(true);
    } catch (error) {
      console.error("Error starting capture:", error);
    }
  }, [intervalSecs, saveDir]);

  const onStopClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/stop`
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      const rawData = await response.json();
      console.log(rawData.file);
      onCaptureStateChange(false);
    } catch (error) {
      console.error("Error stopping capture:", error);
    }
  }, []);

  return (
    <>
      <h2 className="text-center font-bold">Interval Mode</h2>
      <div className="flex flex-row gap-2 items-center">
        <Label htmlFor="interval">Interval (seconds):</Label>
        <Input
          type="number"
          id="interval"
          value={intervalSecs}
          onChange={onIntervalChange}
        />
      </div>
      {captureInProgress ? (
        <Button onClick={onStopClick}>Stop</Button>
      ) : (
        <Button onClick={onStartClick}>Start</Button>
      )}
    </>
  );
}

function OverlapMode({
  saveDir,
  captureInProgress,
  onCaptureStateChange,
}: {
  saveDir: string;
  captureInProgress: boolean;
  onCaptureStateChange: (inProgress: boolean) => void;
}) {
  const [overlap, setOverlap] = useState<number>(50);

  const onOverlapChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(e.target.value);
      if (isNaN(value) || value < 0) {
        return;
      }
      setOverlap(value);
    },
    []
  );

  const onStartClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/overlap`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ overlap, saveDir }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      const rawData = await response.json();
      console.log(rawData.file);
      onCaptureStateChange(true);
    } catch (error) {
      console.error("Error starting capture:", error);
    }
  }, [overlap, saveDir]);

  const onStopClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/stop`
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      const rawData = await response.json();
      console.log(rawData.file);
      onCaptureStateChange(false);
    } catch (error) {
      console.error("Error stopping capture:", error);
    }
  }, []);

  return (
    <>
      <h2 className="text-center font-bold">Overlap Mode</h2>
      <div className="flex flex-row gap-2 items-center">
        <Label htmlFor="overlap">Overlap (%):</Label>
        <Input
          type="number"
          id="overlap"
          value={overlap}
          onChange={onOverlapChange}
        />
      </div>
      {captureInProgress ? (
        <Button onClick={onStopClick}>Stop</Button>
      ) : (
        <Button onClick={onStartClick}>Start</Button>
      )}
    </>
  );
}
