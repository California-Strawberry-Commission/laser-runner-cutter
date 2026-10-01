import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { cn, enumToLabel } from "@/lib/utils";

export enum CalibrationState {
  UNAVAILABLE,
  UNCALIBRATED,
  CALIBRATING,
  CALIBRATED,
}

export type CalibrationStats = {
  numPoints: number;
  depthRange: [number, number];
  meanPositionError: number;
};

// Camera-space depth range (mm) that the calibration should cover
export const REQUIRED_CALIBRATION_DEPTH_RANGE: [number, number] = [500, 1500];
// How far (mm) each end of the calibration depth range may fall short of the
// required range before it is considered insufficient
export const CALIBRATION_DEPTH_RANGE_TOLERANCE = 100;

function formatDepthRange([min, max]: [number, number]) {
  return `${min.toFixed(0)}–${max.toFixed(0)} mm`;
}

export default function CalibrationCard({
  calibrationState,
  calibrationStats,
  disabled,
  onCalibrateClick,
  onStopClick,
  onClearClick,
  onSaveClick,
  onLoadClick,
  className,
}: {
  calibrationState: CalibrationState;
  calibrationStats?: CalibrationStats;
  disabled?: boolean;
  onCalibrateClick?: React.MouseEventHandler<HTMLButtonElement>;
  onStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  onClearClick?: React.MouseEventHandler<HTMLButtonElement>;
  onSaveClick?: React.MouseEventHandler<HTMLButtonElement>;
  onLoadClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  const depthRangeSufficient =
    !calibrationStats ||
    (calibrationStats.depthRange[0] <=
      REQUIRED_CALIBRATION_DEPTH_RANGE[0] + CALIBRATION_DEPTH_RANGE_TOLERANCE &&
      calibrationStats.depthRange[1] >=
        REQUIRED_CALIBRATION_DEPTH_RANGE[1] -
          CALIBRATION_DEPTH_RANGE_TOLERANCE);

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
      cardColor = depthRangeSufficient ? "bg-green-500" : "bg-yellow-400";
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

  const calibrated = calibrationState === CalibrationState.CALIBRATED;
  const stateLabel =
    calibrated && !depthRangeSufficient
      ? "Insufficient calibration depth range"
      : enumToLabel(CalibrationState[calibrationState]);

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4 flex flex-row items-start justify-between gap-4 space-y-0">
        <div className="flex flex-col space-y-1.5">
          <CardTitle className="text-lg">Calibration</CardTitle>
          <CardDescription className="text-foreground">
            {stateLabel}
          </CardDescription>
        </div>
        {calibrated && calibrationStats && (
          <Dialog>
            <DialogTrigger asChild>
              <Button size="sm">Details</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Calibration Details</DialogTitle>
                <DialogDescription>
                  Statistics of the current camera-laser calibration model
                </DialogDescription>
              </DialogHeader>
              <Table>
                <TableBody>
                  <TableRow>
                    <TableCell className="font-medium">
                      Calibration points
                    </TableCell>
                    <TableCell>{calibrationStats.numPoints}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">Depth range</TableCell>
                    <TableCell>
                      {formatDepthRange(calibrationStats.depthRange)}
                      {!depthRangeSufficient && (
                        <span className="block text-yellow-600">
                          {`Does not cover ${formatDepthRange(
                            REQUIRED_CALIBRATION_DEPTH_RANGE,
                          )}`}
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">
                      Mean position error
                    </TableCell>
                    <TableCell>{`${calibrationStats.meanPositionError.toFixed(2)} mm`}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="outline">Close</Button>
                </DialogClose>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </CardHeader>
      <CardContent className="p-4 pt-0 flex flex-row gap-4">
        {calibrateButton}
        <Button
          disabled={
            disabled || calibrationState !== CalibrationState.CALIBRATED
          }
          onClick={onClearClick}
        >
          Clear
        </Button>
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
