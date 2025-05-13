"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { useCallback, useState } from "react";

export enum RunnerCutterState {
  UNAVAILABLE,
  IDLE,
  TRACKING,
  ARMED_AUTO,
  ARMED_MANUAL,
}

export enum RunnerCutterMode {
  TRACKING_ONLY,
  AUTO,
  MANUAL,
}

function enumToLabel(value: string): string {
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

export default function RunnerCutterCard({
  runnerCutterState,
  disabled,
  onStartClick,
  onStopClick,
  className,
}: {
  runnerCutterState: RunnerCutterState;
  disabled?: boolean;
  onStartClick?: (mode: RunnerCutterMode) => void;
  onStopClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}) {
  const [selectedMode, setSelectedMode] = useState<RunnerCutterMode | null>(
    null
  );

  let cardColor = null;
  let startStopButton = null;
  switch (runnerCutterState) {
    case RunnerCutterState.IDLE:
      cardColor = "bg-green-500";
      startStopButton = (
        <Button
          disabled={disabled}
          onClick={() => {
            if (selectedMode !== null && onStartClick) {
              onStartClick(selectedMode);
            }
          }}
        >
          Start
        </Button>
      );
      break;
    case RunnerCutterState.TRACKING:
    case RunnerCutterState.ARMED_MANUAL:
    case RunnerCutterState.ARMED_AUTO:
      cardColor = "bg-yellow-300";
      startStopButton = (
        <Button disabled={disabled} variant="destructive" onClick={onStopClick}>
          Stop
        </Button>
      );
      break;
    default:
      cardColor = "bg-gray-300";
      startStopButton = <Button disabled>Start</Button>;
      break;
  }

  const selector = (
    <Select
      onValueChange={(value) => {
        const mode = RunnerCutterMode[value as keyof typeof RunnerCutterMode];
        setSelectedMode(mode);
      }}
      disabled={disabled || runnerCutterState !== RunnerCutterState.IDLE}
    >
      <SelectTrigger className="w-[150px]">
        <SelectValue placeholder="Select a mode" />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          {Object.keys(RunnerCutterMode)
            .filter((key) => isNaN(Number(key)))
            .map((key) => (
              <SelectItem key={key} value={key}>
                {enumToLabel(key)}
              </SelectItem>
            ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );

  return (
    <Card className={cn(cardColor, className)}>
      <CardHeader className="p-4">
        <CardTitle className="text-lg">Runner Cutter</CardTitle>
        <CardDescription className="text-foreground">
          {enumToLabel(RunnerCutterState[runnerCutterState])}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0 flex flex-row gap-4">
        {selector}
        {startStopButton}
      </CardContent>
    </Card>
  );
}
