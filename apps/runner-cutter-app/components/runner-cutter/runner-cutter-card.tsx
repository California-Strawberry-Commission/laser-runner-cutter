import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export enum RunnerCutterState {
  UNAVAILABLE,
  IDLE,
  TRACKING,
  ARMED,
}

export default function RunnerCutterCard({
  runnerCutterState,
  disabled,
  onTrackClick,
  onTrackStopClick,
  onArmClick,
  onArmStopClick,
  className,
}: {
  runnerCutterState: RunnerCutterState;
  disabled?: boolean;
  onTrackClick?: React.MouseEventHandler<HTMLButtonElement>;
  onTrackStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  onArmClick?: React.MouseEventHandler<HTMLButtonElement>;
  onArmStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  let cardColor = null;
  let trackButton = null;
  let armButton = null;
  switch (runnerCutterState) {
    case RunnerCutterState.IDLE:
      cardColor = "bg-green-500";
      trackButton = (
        <Button disabled={disabled} onClick={onTrackClick}>
          Tracking Only
        </Button>
      );
      armButton = (
        <Button disabled={disabled} onClick={onArmClick}>
          Arm
        </Button>
      );
      break;
    case RunnerCutterState.TRACKING:
      cardColor = "bg-yellow-300";
      trackButton = (
        <Button
          disabled={disabled}
          variant="destructive"
          onClick={onTrackStopClick}
        >
          Stop Tracking
        </Button>
      );
      armButton = <Button disabled>Arm</Button>;
      break;
    case RunnerCutterState.ARMED:
      cardColor = "bg-yellow-300";
      trackButton = <Button disabled>Tracking Only</Button>;
      armButton = (
        <Button
          disabled={disabled}
          variant="destructive"
          onClick={onArmStopClick}
        >
          Stop
        </Button>
      );
      break;
    default:
      cardColor = "bg-gray-300";
      trackButton = <Button disabled>Tracking Only</Button>;
      armButton = <Button disabled>Arm</Button>;
      break;
  }

  let stateStr = RunnerCutterState[runnerCutterState];
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
      </div>
    </Card>
  );
}
