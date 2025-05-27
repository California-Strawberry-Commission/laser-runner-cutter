import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { DeviceState as LaserDeviceState } from "@/lib/useLaserNode";
import { DeviceState as CameraDeviceState } from "@/lib/useCameraNode";
import { cn, enumToLabel } from "@/lib/utils";
import { Loader2 } from "lucide-react";

export enum DeviceState {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
  DISCONNECTING,
  UNAVAILABLE,
}

export function convertLaserNodeDeviceState(laserNode: any): DeviceState {
  let deviceState = DeviceState.UNAVAILABLE;
  if (laserNode.connected) {
    switch (laserNode.state.deviceState) {
      case LaserDeviceState.DISCONNECTED:
        deviceState = DeviceState.DISCONNECTED;
        break;
      case LaserDeviceState.CONNECTING:
        deviceState = DeviceState.CONNECTING;
        break;
      case LaserDeviceState.PLAYING:
      case LaserDeviceState.STOPPED:
        deviceState = DeviceState.CONNECTED;
        break;
      case LaserDeviceState.DISCONNECTING:
        deviceState = DeviceState.DISCONNECTING;
        break;
      default:
        break;
    }
  }
  return deviceState;
}

export function convertCameraNodeDeviceState(cameraNode: any): DeviceState {
  let deviceState = DeviceState.UNAVAILABLE;
  if (cameraNode.connected) {
    switch (cameraNode.state.deviceState) {
      case CameraDeviceState.DISCONNECTED:
        deviceState = DeviceState.DISCONNECTED;
        break;
      case CameraDeviceState.CONNECTING:
        deviceState = DeviceState.CONNECTING;
        break;
      case CameraDeviceState.STREAMING:
        deviceState = DeviceState.CONNECTED;
        break;
      case CameraDeviceState.DISCONNECTING:
        deviceState = DeviceState.DISCONNECTING;
        break;
      default:
        break;
    }
  }
  return deviceState;
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
    case DeviceState.DISCONNECTING:
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

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4">
        <CardTitle className="text-lg">{deviceName}</CardTitle>
        <CardDescription className="text-foreground">
          {enumToLabel(DeviceState[deviceState])}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0">{button}</CardContent>
    </Card>
  );
}
