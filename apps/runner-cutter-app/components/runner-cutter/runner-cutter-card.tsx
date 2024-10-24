import { CalibrationState } from "@/components/runner-cutter/calibration-card";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export default function RunnerCutterCard({
  calibrationState,
  onTrackClick,
  onArmClick,
  onStopClick,
  className,
}: {
  calibrationState: CalibrationState;
  onTrackClick?: React.MouseEventHandler<HTMLButtonElement>;
  onArmClick?: React.MouseEventHandler<HTMLButtonElement>;
  onStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  let cardColor;
  let trackButton = null;
  let armButton = null;
  let stopButton = null;
  switch (calibrationState) {
    case CalibrationState.BUSY:
      cardColor = "bg-gray-300";
      stopButton = (
        <Button variant="destructive" onClick={onStopClick}>
          Stop
        </Button>
      );
      break;
    case CalibrationState.CALIBRATED:
      cardColor = "bg-green-500";
      trackButton = <Button onClick={onTrackClick}>Tracking Only</Button>;
      armButton = <Button onClick={onArmClick}>Arm</Button>;
      break;
    default:
      cardColor = "bg-gray-300";
      trackButton = <Button disabled>Tracking Only</Button>;
      armButton = <Button disabled>Arm</Button>;
      break;
  }

  let stateStr = CalibrationState[calibrationState];
  stateStr = stateStr.charAt(0).toUpperCase() + stateStr.slice(1).toLowerCase();

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4">
        <CardTitle className="text-lg">Runner Cutter</CardTitle>
        <CardDescription className="text-foreground">
          {stateStr}
        </CardDescription>
      </CardHeader>
      <div className="p-4 pt-0 w-full flex flex-row gap-4">
        {trackButton}
        {armButton}
        {stopButton}
      </div>
    </Card>
  );
}
