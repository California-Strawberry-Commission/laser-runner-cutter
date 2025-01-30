import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

export enum CircleFollowerState {
  UNAVAILABLE,
  IDLE,
  TRACKING,
  FOLLOWING,
}

export default function CircleFollowerCard({
  circleFollowerState,
  disabled,
  onTrackClick,
  onTrackStopClick,
  onFollowClick,
  onFollowStopClick,
  className,
}: {
  circleFollowerState: CircleFollowerState;
  disabled?: boolean;
  onTrackClick?: React.MouseEventHandler<HTMLButtonElement>;
  onTrackStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  onFollowClick?: React.MouseEventHandler<HTMLButtonElement>;
  onFollowStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  let cardColor = null;
  let trackButton = null;
  let followButton = null;
  switch (circleFollowerState) {
    case CircleFollowerState.IDLE:
      cardColor = "bg-green-500";
      trackButton = (
        <Button disabled={disabled} onClick={onTrackClick}>
          Tracking Only
        </Button>
      );
      followButton = (
        <Button disabled={disabled} onClick={onFollowClick}>
          Follow
        </Button>
      );
      break;
    case CircleFollowerState.TRACKING:
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
      followButton = <Button disabled>Follow</Button>;
      break;
    case CircleFollowerState.FOLLOWING:
      cardColor = "bg-yellow-300";
      trackButton = <Button disabled>Tracking Only</Button>;
      followButton = (
        <Button
          disabled={disabled}
          variant="destructive"
          onClick={onFollowStopClick}
        >
          Stop
        </Button>
      );
      break;
    default:
      cardColor = "bg-gray-300";
      trackButton = <Button disabled>Tracking Only</Button>;
      followButton = <Button disabled>Follow</Button>;
      break;
  }

  let stateStr = CircleFollowerState[circleFollowerState];
  stateStr = stateStr.charAt(0).toUpperCase() + stateStr.slice(1).toLowerCase();

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4">
        <CardTitle className="text-lg">Circle Follower</CardTitle>
        <CardDescription className="text-foreground">
          {stateStr}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0 flex flex-row gap-4">
        {trackButton}
        {followButton}
      </CardContent>
    </Card>
  );
}
