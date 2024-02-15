import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import KeyboardContext from "@/lib/keyboard-context";

enum CaptureMode {
  MANUAL,
  INTERVAL,
  OVERLAP,
}

export default function Configuration({
  logLimit = 100,
}: {
  logLimit?: number;
}) {
  const webSocket = useRef<WebSocket | null>(null);
  const [saveDir, setSaveDir] = useState<string>("~/Pictures/runners");
  const [filePrefix, setFilePrefix] = useState<string>("runner_");
  const [exposureMs, setExposureMs] = useState<number>(0.2);
  const [captureMode, setCaptureMode] = useState<CaptureMode>(
    CaptureMode.MANUAL
  );
  const [captureInProgress, setCaptureInProgress] = useState<boolean>(false);
  const [logMessages, setLogMessages] = useState<string[]>([]);

  useEffect(() => {
    const startWebSocket = () => {
      const ws = new WebSocket(`ws://${window.location.hostname}:8042/log`);

      ws.onmessage = (event) => {
        const msg = event.data;
        const timestamp = new Date().toLocaleTimeString();
        setLogMessages((prevLogMessages) => {
          const newMessages = [...prevLogMessages, `[${timestamp}] ${msg}`];
          if (newMessages.length > logLimit) {
            newMessages.shift();
          }
          return newMessages;
        });
      };

      ws.onclose = () => {
        setTimeout(startWebSocket, 1000);
      };

      webSocket.current = ws;
    };

    startWebSocket();

    return () => {
      if (webSocket.current) {
        webSocket.current.close();
      }
    };
  }, []);

  const onExposureApplyClick = async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/camera/exposure`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ exposureMs }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      await response.json();
    } catch (error) {
      console.error("Error requesting capture:", error);
    }
  };

  const onModeClick = useCallback(
    (captureMode: CaptureMode) => {
      setCaptureMode(captureMode);
    },
    [setCaptureMode]
  );

  const onCaptureStateChange = useCallback(
    (inProgress: boolean) => {
      setCaptureInProgress(inProgress);
    },
    [setCaptureInProgress]
  );

  return (
    <div className="flex flex-col gap-2 w-96">
      <div className="flex flex-row gap-2 items-center">
        <Label className="flex-none w-20" htmlFor="saveDir">
          Save Directory:
        </Label>
        <Input
          type="text"
          id="saveDir"
          name="saveDir"
          placeholder="Path where images will be saved"
          value={saveDir}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            setSaveDir(e.target.value);
          }}
          keyboardOnChange={(value) => {
            setSaveDir(value);
          }}
        />
      </div>
      <div className="flex flex-row gap-2 items-center">
        <Label className="flex-none w-20" htmlFor="filePrefix">
          File Prefix:
        </Label>
        <Input
          type="text"
          id="filePrefix"
          name="filePrefix"
          placeholder="String to prepend to filenames"
          value={filePrefix}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            setFilePrefix(e.target.value);
          }}
          keyboardOnChange={(value) => {
            setFilePrefix(value);
          }}
        />
      </div>
      <div className="flex flex-row gap-2 items-center">
        <Label className="flex-none w-20" htmlFor="exposure">
          Exposure (ms):
        </Label>
        <Input
          type="number"
          id="exposure"
          name="exposure"
          step={0.01}
          value={exposureMs}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            const value = Number(e.target.value);
            if (isNaN(value)) {
              return;
            }
            setExposureMs(value);
          }}
          keyboardOnChange={(str) => {
            const value = Number(str);
            if (isNaN(value)) {
              return;
            }
            setExposureMs(value);
          }}
        />
        <Button onClick={onExposureApplyClick}>Apply</Button>
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
        </div>
      </div>
      <Separator className="my-2" />
      {captureMode === CaptureMode.MANUAL ? (
        <ManualMode saveDir={saveDir} filePrefix={filePrefix} />
      ) : null}
      {captureMode === CaptureMode.INTERVAL ? (
        <IntervalMode
          saveDir={saveDir}
          filePrefix={filePrefix}
          captureInProgress={captureInProgress}
          onCaptureStateChange={onCaptureStateChange}
        />
      ) : null}
      {captureMode === CaptureMode.OVERLAP ? (
        <OverlapMode
          saveDir={saveDir}
          filePrefix={filePrefix}
          captureInProgress={captureInProgress}
          onCaptureStateChange={onCaptureStateChange}
        />
      ) : null}
      <Separator className="my-2" />
      <h2 className="text-center font-bold">Log Messages</h2>
      <div className="h-[80px] overflow-y-auto">
        <ul>
          {logMessages
            .slice()
            .reverse()
            .map((message, index) => (
              <li className="text-xs" key={index}>
                {message}
              </li>
            ))}
        </ul>
      </div>
    </div>
  );
}

function ManualMode({
  saveDir,
  filePrefix,
}: {
  saveDir: string;
  filePrefix: string;
}) {
  const onCaptureClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/manual`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ saveDir, filePrefix }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      await response.json();
    } catch (error) {
      console.error("Error requesting capture:", error);
    }
  }, [saveDir, filePrefix]);

  return (
    <>
      <h2 className="text-center font-bold">Manual Mode</h2>
      <Button onClick={onCaptureClick}>Capture</Button>
    </>
  );
}

function IntervalMode({
  saveDir,
  filePrefix,
  captureInProgress,
  onCaptureStateChange,
  step = 1.0,
}: {
  saveDir: string;
  filePrefix: string;
  captureInProgress: boolean;
  onCaptureStateChange: (inProgress: boolean) => void;
  step?: number;
}) {
  const [intervalSecs, setIntervalSecs] = useState<number>(5);
  const { setInputName } = useContext(KeyboardContext);

  const onIntervalChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(e.target.value);
      if (isNaN(value) || value < 0) {
        return;
      }
      setIntervalSecs(value);
    },
    [setIntervalSecs]
  );

  const onIntervalDecrementClick = useCallback(() => {
    setIntervalSecs((prev) => Math.round((prev - step) / step) * step);
  }, [step, setIntervalSecs]);

  const onIntervalIncrementClick = useCallback(() => {
    setIntervalSecs((prev) => Math.round((prev + step) / step) * step);
  }, [step, setIntervalSecs]);

  const onStartClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/interval`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ intervalSecs, saveDir, filePrefix }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      await response.json();
      onCaptureStateChange(true);
    } catch (error) {
      console.error("Error starting capture:", error);
    }
  }, [intervalSecs, saveDir, filePrefix, onCaptureStateChange]);

  const onStopClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/stop`
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      await response.json();
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
        <Button onClick={onIntervalDecrementClick}>-</Button>
        <Input
          type="number"
          id="interval"
          name="interval"
          step={step}
          min={0}
          value={intervalSecs}
          onFocus={() => setInputName("interval")}
          onChange={onIntervalChange}
        />
        <Button onClick={onIntervalIncrementClick}>+</Button>
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
  filePrefix,
  captureInProgress,
  onCaptureStateChange,
}: {
  saveDir: string;
  filePrefix: string;
  captureInProgress: boolean;
  onCaptureStateChange: (inProgress: boolean) => void;
}) {
  const [overlap, setOverlap] = useState<number>(50);
  const { setInputName } = useContext(KeyboardContext);

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
          body: JSON.stringify({ overlap, saveDir, filePrefix }),
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      await response.json();
      onCaptureStateChange(true);
    } catch (error) {
      console.error("Error starting capture:", error);
    }
  }, [overlap, saveDir, filePrefix]);

  const onStopClick = useCallback(async () => {
    try {
      const response = await fetch(
        `${window.location.protocol}//${window.location.hostname}:8042/capture/stop`
      );

      if (!response.ok) {
        throw new Error("Network response was not ok " + response.statusText);
      }

      await response.json();
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
          name="overlap"
          value={overlap}
          onFocus={() => setInputName("overlap")}
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
