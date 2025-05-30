"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { RgbColor } from "@/lib/useRgbColor";
import { cn } from "@/lib/utils";
import { useCallback } from "react";
import { RgbColorPicker } from "react-colorful";

function parseNormalizedRgbString(
  rgb: string
): { r: number; g: number; b: number } | null {
  const match = rgb.match(
    /^rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/
  );
  if (!match) {
    return null;
  }

  const clamp = (n: number) => Math.max(0, Math.min(1, n));

  const [, rStr, gStr, bStr] = match;
  return {
    r: clamp(parseFloat(rStr)),
    g: clamp(parseFloat(gStr)),
    b: clamp(parseFloat(bStr)),
  };
}

function normalize(color: RgbColor) {
  return {
    r: color.r / 255,
    g: color.g / 255,
    b: color.b / 255,
  };
}

function denormalize(normalizedColor: RgbColor) {
  return {
    r: Math.round(normalizedColor.r * 255),
    g: Math.round(normalizedColor.g * 255),
    b: Math.round(normalizedColor.b * 255),
  };
}

export default function ColorPicker({
  color,
  onColorChange,
  className,
}: {
  color: RgbColor;
  onColorChange: (color: RgbColor) => void;
  className?: string;
}) {
  const truncate = (n: number) => Math.trunc(n * 100) / 100;
  const colorStr = `rgb(${truncate(color.r)}, ${truncate(color.g)}, ${truncate(
    color.b
  )})`;

  const onColorChangeInternal = useCallback(
    (newColor: RgbColor) => {
      onColorChange(normalize(newColor));
    },
    [onColorChange]
  );

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant={"outline"}
          className={cn(
            "w-full justify-start text-left font-normal",
            !color && "text-muted-foreground",
            className
          )}
        >
          <div className="flex w-full items-center gap-2">
            <div
              className="h-4 w-4 rounded !bg-cover !bg-center transition-all"
              style={{
                background: `rgb(${color.r * 255}, ${color.g * 255}, ${
                  color.b * 255
                })`,
              }}
            ></div>
            <div className="flex-1 truncate">{colorStr}</div>
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="flex flex-col justify-center items-center w-[232px]">
        <RgbColorPicker
          color={denormalize(color)}
          onChange={onColorChangeInternal}
        />
        <Input
          id="custom"
          value={colorStr}
          className="col-span-2 mt-4 h-8"
          onChange={(str) => {
            const rgb = parseNormalizedRgbString(str);
            if (rgb) {
              onColorChange(rgb);
            }
          }}
        />
      </PopoverContent>
    </Popover>
  );
}
