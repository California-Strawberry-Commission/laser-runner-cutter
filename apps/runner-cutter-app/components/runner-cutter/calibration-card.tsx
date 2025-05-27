import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn, enumToLabel } from "@/lib/utils";

export enum CalibrationState {
  UNAVAILABLE,
  UNCALIBRATED,
  CALIBRATING,
  CALIBRATED,
}

export default function CalibrationCard({
  calibrationState,
  disabled,
  onCalibrateClick,
  onStopClick,
  onSaveClick,
  onLoadClick,
  className,
}: {
  calibrationState: CalibrationState;
  disabled?: boolean;
  onCalibrateClick?: React.MouseEventHandler<HTMLButtonElement>;
  onStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  onSaveClick?: React.MouseEventHandler<HTMLButtonElement>;
  onLoadClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  let cardColor;
  let calibrateButton = null;
  switch (calibrationState) {
    case CalibrationState.UNCALIBRATED:
      cardColor = "bg-red-500";
      calibrateButton = (
        <Button disabled={disabled} onClick={onCalibrateClick}>
          Calibrate
        </Button>
      );
      break;
    case CalibrationState.CALIBRATING:
      cardColor = "bg-red-500";
      calibrateButton = (
        <Button disabled={disabled} variant="destructive" onClick={onStopClick}>
          Stop
        </Button>
      );
      break;
    case CalibrationState.CALIBRATED:
      cardColor = "bg-green-500";
      calibrateButton = (
        <Button disabled={disabled} onClick={onCalibrateClick}>
          Calibrate
        </Button>
      );
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
          {enumToLabel(CalibrationState[calibrationState])}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0 flex flex-row gap-4">
        {calibrateButton}
        <Button
          disabled={
            disabled || calibrationState !== CalibrationState.CALIBRATED
          }
          onClick={onSaveClick}
        >
          Save
        </Button>
        <Button
          disabled={
            disabled ||
            calibrationState === CalibrationState.UNAVAILABLE ||
            calibrationState === CalibrationState.CALIBRATING
          }
          onClick={onLoadClick}
        >
          Load
        </Button>
      </CardContent>
    </Card>
  );
}
