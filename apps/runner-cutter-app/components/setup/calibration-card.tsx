import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export enum CalibrationState {
  Uncalibrated,
  Busy,
  Calibrated,
  Unavailable,
}

export default function DeviceCard({
  calibrationState,
  onCalibrateClick,
  onStopClick,
  onSaveClick,
  onLoadClick,
  className,
}: {
  calibrationState: CalibrationState;
  onCalibrateClick?: React.MouseEventHandler<HTMLButtonElement>;
  onStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  onSaveClick?: React.MouseEventHandler<HTMLButtonElement>;
  onLoadClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  let cardColor;
  let calibrateButton = null;
  switch (calibrationState) {
    case CalibrationState.Uncalibrated:
      cardColor = "bg-red-500";
      calibrateButton = <Button onClick={onCalibrateClick}>Calibrate</Button>;
      break;
    case CalibrationState.Busy:
      cardColor = "bg-gray-300";
      calibrateButton = (
        <Button variant="destructive" onClick={onStopClick}>
          Stop
        </Button>
      );
      break;
    case CalibrationState.Calibrated:
      cardColor = "bg-green-500";
      calibrateButton = <Button onClick={onCalibrateClick}>Calibrate</Button>;
      break;
    default:
      cardColor = "bg-gray-300";
      calibrateButton = <Button disabled>Calibrate</Button>;
      break;
  }

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4">
        <CardTitle className="text-lg">Calibration</CardTitle>
        <CardDescription className="text-foreground">
          {CalibrationState[calibrationState]}
        </CardDescription>
      </CardHeader>
      <div className="p-4 pt-0 w-full flex flex-row gap-4">
        {calibrateButton}
        <Button
          disabled={calibrationState !== CalibrationState.Calibrated}
          onClick={onSaveClick}
        >
          Save
        </Button>
        <Button
          disabled={
            calibrationState === CalibrationState.Unavailable ||
            calibrationState === CalibrationState.Busy
          }
          onClick={onLoadClick}
        >
          Load
        </Button>
      </div>
    </Card>
  );
}
