import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

export enum DeviceState {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
  UNAVAILABLE,
}

export default function DeviceCard({
  deviceName,
  deviceState,
  onConnectClick,
  onDisconnectClick,
  className,
}: {
  deviceName: string;
  deviceState: DeviceState;
  onConnectClick?: React.MouseEventHandler<HTMLButtonElement>;
  onDisconnectClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  let cardColor;
  let button = null;
  switch (deviceState) {
    case DeviceState.DISCONNECTED:
      cardColor = "bg-red-500";
      button = (
        <Button className="w-full" onClick={onConnectClick}>
          Connect
        </Button>
      );
      break;
    case DeviceState.CONNECTING:
      cardColor = "bg-gray-300";
      button = (
        <Button className="w-full" disabled>
          <Loader2 className="h-4 w-4 animate-spin" />
        </Button>
      );
      break;
    case DeviceState.CONNECTED:
      cardColor = "bg-green-500";
      button = (
        <Button className="w-full" onClick={onDisconnectClick}>
          Disconnect
        </Button>
      );
      break;
    default:
      cardColor = "bg-gray-300";
      button = (
        <Button className="w-full" disabled>
          Unavailable
        </Button>
      );
      break;
  }

  let stateStr = DeviceState[deviceState];
  stateStr = stateStr.charAt(0).toUpperCase() + stateStr.slice(1).toLowerCase();

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4">
        <CardTitle className="text-lg">{deviceName}</CardTitle>
        <CardDescription className="text-foreground">
          {stateStr}
        </CardDescription>
      </CardHeader>
      <div className="p-4 pt-0 w-full">{button}</div>
    </Card>
  );
}
